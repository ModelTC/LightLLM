from __future__ import annotations

import math
import os
import random
from collections import Counter, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from sortedcontainers import SortedDict

from lightllm.utils.log_utils import init_logger


logger = init_logger(__name__)


@dataclass(frozen=True)
class SpecDecodePlan:
    """Planner decision for one target decode iteration.

    Static MTP uses the full MTP-expanded target batch:
    - dynamic_batch_size is None
    - draft_step == mtp_step

    Dynamic MTP may compact target rows before forward:
    - dynamic_batch_size is the selected target row count
    - draft_step is the candidate length to generate after target verify
    - pre_draft_step describes the previous iteration and controls whether
      GPU verify sync can be skipped
    """

    dynamic_batch_size: Optional[int]
    draft_step: int
    pre_draft_step: int

    @property
    def is_dynamic(self) -> bool:
        return self.dynamic_batch_size is not None

    @property
    def skip_verify_sync(self) -> bool:
        return self.is_dynamic and self.pre_draft_step == 0


class FixedMTPPlanner:
    """Planner for static MTP."""

    def __init__(self, mtp_step: int) -> None:
        self.mtp_step = int(mtp_step)

    def plan(self, req_num: int | None = None, original_batch_size: int | None = None) -> SpecDecodePlan:
        del req_num
        del original_batch_size
        return SpecDecodePlan(
            dynamic_batch_size=None,
            draft_step=self.mtp_step,
            pre_draft_step=self.mtp_step,
        )


class DynamicMTPPlanner:
    planner_mode = "default"

    def __init__(
        self,
        mtp_step: int,
        use_random_mode: bool = True,
        random_mode_iter_threshold: int = 100,
    ) -> None:
        self.mtp_step = int(mtp_step)

        # 用于记录 decode 时的静态推理耗时(ms)。
        self.main_model_speeds = _InferCostMsTable()
        self.draft_model_speeds = _InferCostMsTable()

        # 记录每个对应长度mtp step 步的接受概率。 由于原始位置必然是接受的，所以不需要记录。
        self.mtp_len_to_accept_ratio = [
            _EMAValue(decay=0.95, init_value=1.0, enable_decay_warmup=False) for _ in range(self.mtp_step)
        ]
        # 记录请求数量以及对应的推理dynamic_batch_size 对应的接受率统计
        self.req_num_to_dynamic_batch_size_to_accept_ratio: Dict[int, Dict[int, _EMAValue]] = {}

        # 每多少个请求采用随机的方式决定 dynamic_batch_size
        self._iter = 0
        self._iter_threshold = int(random_mode_iter_threshold)
        self._use_random_mode = bool(use_random_mode)
        random.seed(0)

        # 记录上一次选择的draft step 步长，才好选择对应的 dynamic_batch_size
        self.pre_draft_step = self.mtp_step
        return

    def plan(self, req_num: int, original_batch_size: int) -> SpecDecodePlan:
        dynamic_batch_size, draft_step, pre_draft_step = self.get_dynamic_batch_size(
            req_num=req_num,
            original_batch_size=original_batch_size,
        )
        return SpecDecodePlan(
            dynamic_batch_size=dynamic_batch_size,
            draft_step=draft_step,
            pre_draft_step=pre_draft_step,
        )

    def update_infer_cost(self, *, batch_size: int, infer_cost_ms: float, is_draft_model: bool) -> None:
        speed_table = self.draft_model_speeds if is_draft_model else self.main_model_speeds
        speed_table.update(batch_size=batch_size, infer_cost_ms=infer_cost_ms)
        return

    def update_mtp_len_to_accept_ratio(self, mtp_len: int, accept_ratio: float) -> None:
        assert mtp_len > 0 and mtp_len <= self.mtp_step
        self.mtp_len_to_accept_ratio[mtp_len - 1].update(accept_ratio)
        return

    def update_verified_prefix_stats(self, *, verify_len: int, accept_len: int) -> None:
        if verify_len - 1 <= 0:
            return
        for mtp_index in range(verify_len - 1):
            mtp_len = mtp_index + 1
            ratio = (accept_len - 1) / mtp_len
            ratio = max(0.0, min(1.0, ratio))
            self.update_mtp_len_to_accept_ratio(
                mtp_len=mtp_len,
                accept_ratio=ratio,
            )
        return

    def update_req_num_to_dynamic_batch_size_to_accept_ratio(
        self, req_num: int, dynamic_batch_size: int, accept_ratio: float
    ) -> None:
        assert dynamic_batch_size >= req_num
        self._get_req_num_to_dynamic_batch_size_to_accept_ratio(
            req_num=req_num, dynamic_batch_size=dynamic_batch_size
        ).update(accept_ratio)
        return

    def _get_req_num_to_dynamic_batch_size_to_accept_ratio(self, req_num: int, dynamic_batch_size: int) -> "_EMAValue":
        assert dynamic_batch_size >= req_num
        if req_num not in self.req_num_to_dynamic_batch_size_to_accept_ratio:
            self.req_num_to_dynamic_batch_size_to_accept_ratio[req_num] = {}
        if dynamic_batch_size not in self.req_num_to_dynamic_batch_size_to_accept_ratio[req_num]:
            self.req_num_to_dynamic_batch_size_to_accept_ratio[req_num][dynamic_batch_size] = _EMAValue(
                decay=0.9, init_value=1.0, enable_decay_warmup=True
            )
        return self.req_num_to_dynamic_batch_size_to_accept_ratio[req_num][dynamic_batch_size]

    def get_dynamic_batch_size(self, req_num: int, original_batch_size: int) -> Tuple[int, int, int]:
        """
        返回 (dynamic_batch_size, draft_step, pre_draft_step)。
        pre_draft_step 是上一轮推理实际使用的 draft_step，
        调用方可据此判断当前 verify 是否有真实候选需要验证
        (pre_draft_step == 0 时 accept_len 恒为 1，无需等待 GPU verify 结果)。
        """
        assert req_num * (self.mtp_step + 1) == original_batch_size
        pre_draft_step = self.pre_draft_step
        if req_num == 0:
            self.pre_draft_step = self.mtp_step
            return 0, self.mtp_step, pre_draft_step
        if not self.main_model_speeds.has_data() or not self.draft_model_speeds.has_data():
            # The cost model is only meaningful after both target and draft
            # decode costs have been profiled.  Block proposers such as DFlash
            # do not run through draft_model.forward, and cudagraph may also be
            # disabled, so a missing table must not collapse dynamic MTP to
            # draft_step=0.
            self.pre_draft_step = self.mtp_step
            return req_num * (pre_draft_step + 1), self.mtp_step, pre_draft_step

        # case 1 如果采用随机的方式决定 dynamic_batch_size
        self._iter += 1
        if self._use_random_mode and self._iter % self._iter_threshold == 0:
            min_batch_size = req_num
            max_batch_size = req_num * (pre_draft_step + 1)
            dynamic_batch_size = random.randint(min_batch_size, max_batch_size)

            draft_step = random.randint(0, self.mtp_step)
            self.pre_draft_step = draft_step
            return dynamic_batch_size, draft_step, pre_draft_step

        # 通过计算的方式来获取已经知道的最优的 dynamic_batch_size，然后再决定 draft step 步长
        min_batch_size = req_num
        max_batch_size = req_num * (pre_draft_step + 1)
        dynamic_batch_size_keys = self.main_model_speeds.get_batch_size_keys_between(min_batch_size, max_batch_size)

        # 计算每个 dynamic_batch_size 对应的接受率以及单token的速度收益，然后选择最优的 dynamic_batch_size
        cost_ms_list = [
            self._get_cost_ms(req_num=req_num, dynamic_batch_size=dynamic_batch_size, draft_step=pre_draft_step)
            for dynamic_batch_size in dynamic_batch_size_keys
        ]
        dynamic_batch_size = dynamic_batch_size_keys[np.argmin(cost_ms_list)]

        # 下一步的 draft step 选择，需要考虑计算不同step步的收益问题再决定
        min_cost_ms = float("inf")
        min_cost_ms_draft_step = 0  # 默认选择0步长
        for draft_step in range(0, self.mtp_step + 1):
            cost_ms = self._get_cost_ms(req_num=req_num, dynamic_batch_size=dynamic_batch_size, draft_step=draft_step)
            if cost_ms < min_cost_ms:
                min_cost_ms = cost_ms
                min_cost_ms_draft_step = draft_step

        # draft step 步长不能超过 mtp_step, 也不能小于0
        min_cost_ms_draft_step = min(min_cost_ms_draft_step, self.mtp_step)
        min_cost_ms_draft_step = max(min_cost_ms_draft_step, 0)
        self.pre_draft_step = min_cost_ms_draft_step
        return dynamic_batch_size, min_cost_ms_draft_step, pre_draft_step

    def _get_cost_ms(self, req_num: int, dynamic_batch_size: int, draft_step: int) -> float:
        accept_ratio = self._get_dynamic_batch_size_to_accept_ratio(
            req_num=req_num, dynamic_batch_size=dynamic_batch_size
        )
        total_time = (
            self.main_model_speeds.get(dynamic_batch_size)
            + self.draft_model_speeds.get(dynamic_batch_size) * draft_step
        )
        token_num = min((dynamic_batch_size * accept_ratio), req_num * (draft_step + 1))
        token_num = max(token_num, req_num)
        cost_ms = total_time / token_num
        return cost_ms

    def _get_dynamic_batch_size_to_accept_ratio(self, req_num: int, dynamic_batch_size: int):
        ema = self._get_req_num_to_dynamic_batch_size_to_accept_ratio(
            req_num=req_num, dynamic_batch_size=dynamic_batch_size
        )
        if ema.get_count() >= 10:
            # 当以及通过充分的数据统计以后，直接返回统计的接受率
            return ema.get()

        # 通过单请求的信息进行估计。
        real_step = dynamic_batch_size / req_num
        assert real_step >= 1.0
        real_step = real_step - 1.0

        # 用插值的方式估计不同mtp_len 对应的接受率
        left = int(math.floor(real_step))
        right = int(left + 1)
        if left == 0:
            left_value = 0.0
        else:
            left_value = self.mtp_len_to_accept_ratio[left - 1].get()

        if right > self.mtp_step:
            right_value = 0.0
        else:
            right_value = self.mtp_len_to_accept_ratio[right - 1].get()

        accept_ratio = left_value + (right_value - left_value) * (real_step - left)
        calcu_accept_ratio = (req_num + (dynamic_batch_size - req_num) * accept_ratio) / dynamic_batch_size
        weight = ema.get_count() / 10
        # 通过统计数据和单请求数据进行加权平均，得到最终的接受率
        return calcu_accept_ratio * (1 - weight) + ema.get() * weight


class Eagle3DynamicMTPPlanner(DynamicMTPPlanner):
    """Joint draft-length and verify-capacity planner for Eagle3.

    ``pre_draft_step`` bounds the proposal that is being verified now, while
    ``draft_step`` controls the proposal built for the next target iteration.
    Treating those as the same iteration makes ``draft_step == 0`` an
    absorbing state.  We instead choose the next draft length from a
    steady-state search, then choose the current verify capacity within the
    proposal width that is actually available now.

    Eagle3 also has an asymmetric draft cost: its first forward commits all K
    selected target rows, while every recurrent forward after that processes
    one accepted tail per request (batch B).
    """

    planner_mode = "eagle3"
    _ACCEPT_RATIO_BUCKETS_PER_DRAFT_ROW = 8

    def __init__(self, mtp_step: int) -> None:
        super().__init__(mtp_step=mtp_step, use_random_mode=False)
        # Eagle uses these values as a full-verify survival curve.  Start from
        # the first batch mean instead of decaying slowly from an all-accepted
        # prior; otherwise a 32-iteration calibration still substantially
        # overestimates short draft depths.
        self.mtp_len_to_accept_ratio = [
            _EMAValue(decay=0.95, init_value=1.0, enable_decay_warmup=True) for _ in range(self.mtp_step)
        ]
        self._min_static_progress_ratio = float(os.getenv("LIGHTLLM_EAGLE3_MIN_STATIC_PROGRESS_RATIO", "0.85"))
        # This is the externally visible verify efficiency:
        # accepted target rows / selected target verify rows.  It includes the
        # guaranteed first row, matching the project acceptance reported by
        # the HTTP metrics and benchmark helper.
        self._min_project_accept_ratio = float(
            os.getenv(
                "LIGHTLLM_EAGLE3_MIN_PROJECT_ACCEPT_RATIO",
                # Keep a control margin above the externally requested 80%
                # acceptance.  Stop sequences can discard already-verified
                # tail tokens, so HTTP output/verify metrics are about two
                # points below the planner's model-accepted/verify feedback
                # on GSM8K.
                os.getenv("LIGHTLLM_EAGLE3_MIN_DRAFT_ACCEPT_RATIO", "0.860"),
            )
        )
        self._full_verify_warmup_steps = max(
            0,
            int(os.getenv("LIGHTLLM_EAGLE3_FULL_VERIFY_WARMUP_STEPS", "32")),
        )
        self._full_verify_interval = max(
            0,
            int(os.getenv("LIGHTLLM_EAGLE3_FULL_VERIFY_INTERVAL", "128")),
        )
        self._progress_relax_ratio = float(os.getenv("LIGHTLLM_EAGLE3_PROGRESS_RELAX_RATIO", "1.0"))
        self._capacity_accept_ratio_floor = float(os.getenv("LIGHTLLM_EAGLE3_CAPACITY_ACCEPT_RATIO_FLOOR", "0.80"))
        self._capacity_feedback_gain = float(os.getenv("LIGHTLLM_EAGLE3_CAPACITY_FEEDBACK_GAIN", "0.10"))
        self._capacity_feedback_reference_req_num = max(
            1,
            int(os.getenv("LIGHTLLM_EAGLE3_CAPACITY_FEEDBACK_REFERENCE_REQ_NUM", "128")),
        )
        self._align_verify_rows_to_graph = os.getenv("LIGHTLLM_EAGLE3_ALIGN_VERIFY_ROWS_TO_GRAPH", "1").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self._max_dynamic_draft_step = max(
            0,
            min(
                self.mtp_step,
                int(os.getenv("LIGHTLLM_EAGLE3_MAX_DYNAMIC_DRAFT_STEP", str(self.mtp_step))),
            ),
        )
        assert 0.0 < self._min_static_progress_ratio <= 1.0
        assert 0.0 < self._min_project_accept_ratio <= 1.0
        assert 0.0 < self._progress_relax_ratio <= 1.0
        assert 0.0 < self._capacity_accept_ratio_floor <= 1.0
        assert 0.0 < self._capacity_feedback_gain <= 1.0

        # Exact (B, K) statistics are sparse because the live request batch B
        # changes constantly.  Pool observations by normalized selected draft
        # rows per request so adjacent concurrency levels share evidence.
        self._accept_ratio_by_depth_and_width_bucket: Dict[Tuple[int, int], _EMAValue] = {}
        self._accept_ratio_by_depth_req_and_batch: Dict[Tuple[int, int, int], _EMAValue] = {}

        # A few full-width verifies provide an unbiased estimate of the static
        # Eagle acceptance length.  Dynamic top-K observations alone are
        # intentionally biased toward high-confidence rows and cannot serve as
        # a static baseline.
        self._full_verify_tokens_per_req_value = float(self.mtp_step + 1)
        self._full_verify_accepted_token_sum = 0.0
        self._full_verify_request_count = 0
        self._full_verify_update_count = 0
        self._full_probe_pending = False

        # This feedback comes from real non-full dynamic iterations.  It
        # provides a conservative capacity floor when a sparse cost-table
        # candidate has an over-optimistic expected-token estimate.
        self._observed_dynamic_draft_accept_ratio = _EMAValue(
            decay=0.9,
            init_value=0.75,
            enable_decay_warmup=True,
        )
        self._observed_dynamic_project_accept_ratio = _EMAValue(
            decay=0.9,
            init_value=self._min_project_accept_ratio,
            enable_decay_warmup=True,
        )
        self._observed_dynamic_tokens_per_req = _EMAValue(
            decay=0.8,
            init_value=1.0,
            enable_decay_warmup=True,
        )
        # Acceptance is aggregated with request weights.  An iteration with a
        # single long-tail request must not have the same influence as an
        # iteration with 256 live requests.
        self._observed_dynamic_accepted_token_sum = 0.0
        self._observed_dynamic_verify_row_sum = 0.0
        self._observed_dynamic_request_count = 0

        # Closed-loop verify capacity.  The initial value comes from the
        # unbiased full-width baseline.  Real dynamic iterations then move it
        # toward the largest width that still satisfies project acceptance.
        self._target_verify_rows_per_req_value: Optional[float] = None

        self._plan_log_interval = max(0, int(os.getenv("LIGHTLLM_EAGLE3_PLAN_LOG_INTERVAL", "0")))
        self._plan_count = 0
        self._draft_step_counts = Counter()
        self._verify_rows_per_req_sum = 0.0
        self._expected_tokens_per_req_sum = 0.0

    def update_verified_prefix_stats(self, *, verify_len: int, accept_len: int) -> None:
        """Record the survival probability of each Eagle draft position.

        The generic planner records ``(accept_len - 1) / depth``.  That value
        is neither a conditional probability nor a survival probability and
        can increase with depth.  Eagle's expected-token model needs the
        probability that a token at each depth is actually reached.
        """

        max_mtp_len = min(max(0, verify_len - 1), self.mtp_step)
        for mtp_len in range(1, max_mtp_len + 1):
            self.update_mtp_len_to_accept_ratio(
                mtp_len=mtp_len,
                accept_ratio=1.0 if accept_len > mtp_len else 0.0,
            )

    def update_verified_batch_prefix_stats(
        self,
        *,
        verify_and_accept_lengths: List[Tuple[int, int]],
    ) -> None:
        """Update each depth once with a request-weighted batch mean."""

        for mtp_len in range(1, self.mtp_step + 1):
            eligible_accept_lengths = [
                accept_len for verify_len, accept_len in verify_and_accept_lengths if verify_len > mtp_len
            ]
            if not eligible_accept_lengths:
                continue
            survival_ratio = sum(accept_len > mtp_len for accept_len in eligible_accept_lengths) / len(
                eligible_accept_lengths
            )
            self.update_mtp_len_to_accept_ratio(
                mtp_len=mtp_len,
                accept_ratio=survival_ratio,
            )

    def update_req_num_to_dynamic_batch_size_to_accept_ratio(
        self,
        req_num: int,
        dynamic_batch_size: int,
        accept_ratio: float,
        verify_step: int = None,
    ) -> None:
        depth = self.mtp_step if verify_step is None else int(verify_step)
        exact_key = (depth, int(req_num), int(dynamic_batch_size))
        if exact_key not in self._accept_ratio_by_depth_req_and_batch:
            self._accept_ratio_by_depth_req_and_batch[exact_key] = _EMAValue(
                decay=0.9,
                init_value=1.0,
                enable_decay_warmup=True,
            )
        self._accept_ratio_by_depth_req_and_batch[exact_key].update(accept_ratio)
        self._get_width_bucket_accept_ratio(
            req_num=req_num,
            dynamic_batch_size=dynamic_batch_size,
            verify_step=verify_step,
        ).update(accept_ratio)

    def update_full_verify_tokens_per_req(self, tokens_per_req: float, req_num: int = 1) -> None:
        tokens_per_req = max(1.0, min(float(self.mtp_step + 1), float(tokens_per_req)))
        req_num = max(1, int(req_num))
        self._full_verify_accepted_token_sum += tokens_per_req * req_num
        self._full_verify_request_count += req_num
        self._full_verify_tokens_per_req_value = self._full_verify_accepted_token_sum / self._full_verify_request_count
        self._full_verify_update_count += 1

    def update_observed_iteration_stats(
        self,
        *,
        tokens_per_req: float,
        verify_rows_per_req: float,
        is_full_verify: bool,
        req_num: int = 1,
    ) -> None:
        if is_full_verify or verify_rows_per_req <= 1.0:
            return
        req_num = max(1, int(req_num))
        draft_accept_ratio = (tokens_per_req - 1.0) / (verify_rows_per_req - 1.0)
        draft_accept_ratio = max(0.0, min(1.0, draft_accept_ratio))
        project_accept_ratio = max(0.0, min(1.0, tokens_per_req / verify_rows_per_req))
        self._observed_dynamic_draft_accept_ratio.update(draft_accept_ratio)
        self._observed_dynamic_project_accept_ratio.update(project_accept_ratio)
        self._observed_dynamic_tokens_per_req.update(tokens_per_req)
        self._observed_dynamic_accepted_token_sum += tokens_per_req * req_num
        self._observed_dynamic_verify_row_sum += verify_rows_per_req * req_num
        self._observed_dynamic_request_count += req_num

        self._update_target_verify_rows_per_req(
            tokens_per_req=tokens_per_req,
            verify_rows_per_req=verify_rows_per_req,
            req_num=req_num,
        )

    def _update_target_verify_rows_per_req(
        self,
        *,
        tokens_per_req: float,
        verify_rows_per_req: float,
        req_num: int,
    ) -> None:
        current_target = self._get_controlled_verify_rows_per_req()

        # Holding the accepted-token count locally constant, this is the
        # verify width that lands exactly on the project-acceptance target.
        sample_target = tokens_per_req / self._min_project_accept_ratio

        # If progress is below its static-relative floor, acceptance and
        # progress constraints conflict.  Widen enough to recover progress;
        # subsequent observations will pull the controller back once the
        # additional rows stop paying for themselves.
        min_expected_tokens = self._get_min_expected_tokens_per_req()
        if self._get_observed_tokens_per_req() < min_expected_tokens:
            sample_target = max(
                sample_target,
                verify_rows_per_req * min_expected_tokens / max(tokens_per_req, 1e-6),
            )

        sample_target = max(1.0, min(float(self.mtp_step + 1), sample_target))
        # Bound a single observation.  This is especially important while a
        # batch drains and only a handful of unusually hard requests remain.
        sample_target = max(current_target - 0.5, min(current_target + 0.5, sample_target))
        request_weight = req_num / self._capacity_feedback_reference_req_num
        alpha = 1.0 - (1.0 - self._capacity_feedback_gain) ** request_weight
        self._target_verify_rows_per_req_value = current_target * (1.0 - alpha) + sample_target * alpha

    def _get_observed_project_accept_ratio(self) -> float:
        if self._observed_dynamic_verify_row_sum <= 0.0:
            return self._min_project_accept_ratio
        return self._observed_dynamic_accepted_token_sum / self._observed_dynamic_verify_row_sum

    def _get_observed_draft_accept_ratio(self) -> float:
        conditional_rows = self._observed_dynamic_verify_row_sum - self._observed_dynamic_request_count
        if conditional_rows <= 0.0:
            return self._observed_dynamic_draft_accept_ratio.get()
        conditional_tokens = self._observed_dynamic_accepted_token_sum - self._observed_dynamic_request_count
        return max(0.0, min(1.0, conditional_tokens / conditional_rows))

    def _get_observed_tokens_per_req(self) -> float:
        if self._observed_dynamic_request_count <= 0:
            return self._observed_dynamic_tokens_per_req.get()
        return self._observed_dynamic_accepted_token_sum / self._observed_dynamic_request_count

    def _get_dynamic_batch_size_to_accept_ratio(
        self,
        req_num: int,
        dynamic_batch_size: int,
        verify_step: int = None,
    ):
        depth = self.mtp_step if verify_step is None else int(verify_step)
        exact_key = (depth, int(req_num), int(dynamic_batch_size))
        exact_ema = self._accept_ratio_by_depth_req_and_batch.get(exact_key)
        base_estimate = super()._get_dynamic_batch_size_to_accept_ratio(
            req_num=req_num,
            dynamic_batch_size=dynamic_batch_size,
        )
        if exact_ema is not None and exact_ema.get_count() >= 10:
            return exact_ema.get()

        width_ema = self._get_width_bucket_accept_ratio(
            req_num=req_num,
            dynamic_batch_size=dynamic_batch_size,
            verify_step=verify_step,
        )
        width_weight = min(1.0, width_ema.get_count() / 10.0)
        return base_estimate * (1.0 - width_weight) + width_ema.get() * width_weight

    def _get_width_bucket_accept_ratio(
        self,
        *,
        req_num: int,
        dynamic_batch_size: int,
        verify_step: int = None,
    ) -> "_EMAValue":
        selected_draft_rows_per_req = max(0.0, dynamic_batch_size / req_num - 1.0)
        width_bucket = int(round(selected_draft_rows_per_req * self._ACCEPT_RATIO_BUCKETS_PER_DRAFT_ROW))
        depth = self.mtp_step if verify_step is None else int(verify_step)
        bucket = (depth, width_bucket)
        if bucket not in self._accept_ratio_by_depth_and_width_bucket:
            self._accept_ratio_by_depth_and_width_bucket[bucket] = _EMAValue(
                decay=0.9,
                init_value=1.0,
                enable_decay_warmup=True,
            )
        return self._accept_ratio_by_depth_and_width_bucket[bucket]

    def get_dynamic_batch_size(self, req_num: int, original_batch_size: int) -> Tuple[int, int, int]:
        assert req_num * (self.mtp_step + 1) == original_batch_size
        pre_draft_step = self.pre_draft_step

        if req_num == 0:
            self.pre_draft_step = self.mtp_step
            return 0, self.mtp_step, pre_draft_step

        max_batch_size = req_num * (pre_draft_step + 1)
        if not self.main_model_speeds.has_data() or not self.draft_model_speeds.has_data():
            self.pre_draft_step = self.mtp_step
            return max_batch_size, self.mtp_step, pre_draft_step

        if self._should_schedule_full_probe():
            self._full_probe_pending = True

        force_full_verify = self._should_force_full_verify(pre_draft_step=pre_draft_step)
        if force_full_verify:
            # Keep drafting full width during initial calibration.  A periodic
            # probe may return to the normal dynamic draft length immediately
            # after its one full target verify.
            draft_step = (
                self.mtp_step
                if self._full_verify_update_count < self._full_verify_warmup_steps
                else self._select_next_draft_step(req_num=req_num)
            )
            dynamic_batch_size = max_batch_size
            self.pre_draft_step = draft_step
            expected_tokens_per_req = (
                self._estimate_expected_token_num(
                    req_num=req_num,
                    dynamic_batch_size=dynamic_batch_size,
                    verify_step=pre_draft_step,
                )
                / req_num
            )
            self._record_plan_stats(
                req_num=req_num,
                dynamic_batch_size=dynamic_batch_size,
                draft_step=draft_step,
                expected_tokens_per_req=expected_tokens_per_req,
            )
            return dynamic_batch_size, draft_step, pre_draft_step

        # Once the full-width baseline is calibrated, choose draft depth by
        # the target+draft cost model, but control verify width with real
        # project-acceptance feedback.  Sparse (B, K) cost/acceptance buckets
        # are too optimistic for unseen widths and previously widened GSM8K
        # from about 4.5 to 6 rows/request before converging.
        if self._full_verify_request_count > 0:
            draft_step = self._select_next_draft_step(req_num=req_num)
            target_verify_rows_per_req = self._get_controlled_verify_rows_per_req()
            dynamic_batch_size = int(math.ceil(req_num * target_verify_rows_per_req))
            if self._align_verify_rows_to_graph:
                # Replay already pads an arbitrary target batch to the next
                # captured shape.  Turn those paid-for padding rows into real
                # candidates so they can improve accepted progress for the
                # same target-model graph cost.
                graph_batch_size = self.main_model_speeds.get_ceil_batch_size(
                    dynamic_batch_size,
                    max_batch_size=max_batch_size,
                )
                if graph_batch_size is not None:
                    dynamic_batch_size = graph_batch_size
            dynamic_batch_size = min(max(dynamic_batch_size, req_num), max_batch_size)
            self.pre_draft_step = draft_step
            expected_tokens_per_req = (
                self._estimate_expected_token_num(
                    req_num=req_num,
                    dynamic_batch_size=dynamic_batch_size,
                    verify_step=pre_draft_step,
                )
                / req_num
            )
            self._record_plan_stats(
                req_num=req_num,
                dynamic_batch_size=dynamic_batch_size,
                draft_step=draft_step,
                expected_tokens_per_req=expected_tokens_per_req,
            )
            return dynamic_batch_size, draft_step, pre_draft_step

        # The action selected here builds the proposal consumed by the next
        # target iteration.  Search it independently of the current proposal
        # width so a zero-step iteration can recover on its own.
        draft_step = self._select_next_draft_step(req_num=req_num)

        dynamic_batch_size_keys = self._get_candidate_batch_sizes(
            req_num=req_num,
            max_batch_size=max_batch_size,
        )
        candidates = [
            self._get_eagle3_candidate(
                req_num=req_num,
                dynamic_batch_size=dynamic_batch_size,
                verify_step=pre_draft_step,
                draft_step=draft_step,
            )
            for dynamic_batch_size in dynamic_batch_size_keys
        ]
        best_candidate = self._select_best_candidate(candidates, req_num=req_num)
        dynamic_batch_size = best_candidate[1]
        expected_tokens_per_req = best_candidate[2]
        self.pre_draft_step = draft_step
        self._record_plan_stats(
            req_num=req_num,
            dynamic_batch_size=dynamic_batch_size,
            draft_step=draft_step,
            expected_tokens_per_req=expected_tokens_per_req,
        )
        return dynamic_batch_size, draft_step, pre_draft_step

    def _get_target_verify_rows_per_req(self) -> float:
        min_expected_tokens = self._get_min_expected_tokens_per_req()
        if min_expected_tokens <= 1.0:
            return 1.0
        return min(
            float(self.mtp_step + 1),
            min_expected_tokens / self._min_project_accept_ratio,
        )

    def _get_controlled_verify_rows_per_req(self) -> float:
        if self._target_verify_rows_per_req_value is None:
            return self._get_target_verify_rows_per_req()
        return max(
            1.0,
            min(float(self.mtp_step + 1), self._target_verify_rows_per_req_value),
        )

    def _get_candidate_batch_sizes(self, *, req_num: int, max_batch_size: int) -> List[int]:
        candidates = set(
            self.main_model_speeds.get_batch_size_keys_between(
                req_num,
                max_batch_size,
            )
        )
        candidates.add(req_num)
        candidates.add(max_batch_size)
        exact_target = int(math.ceil(req_num * self._get_target_verify_rows_per_req()))
        if req_num <= exact_target <= max_batch_size:
            candidates.add(exact_target)

        # Add exact points along the feasible progress/acceptance frontier.
        # CUDA graph timing keys are deliberately coarse; without these
        # points the cost search can only choose between two widely separated
        # capacities and often leaves useful acceptance headroom unused.
        if self._full_verify_request_count > 0:
            for progress_ratio in np.linspace(self._min_static_progress_ratio, 1.0, num=7):
                expected_tokens_per_req = self._full_verify_tokens_per_req_value * float(progress_ratio)
                verify_rows_per_req = expected_tokens_per_req / self._min_project_accept_ratio
                frontier_batch_size = int(math.ceil(req_num * verify_rows_per_req))
                if req_num <= frontier_batch_size <= max_batch_size:
                    candidates.add(frontier_batch_size)
        return sorted(candidates)

    def _should_schedule_full_probe(self) -> bool:
        return (
            self._full_verify_interval > 0
            and self._full_verify_update_count >= self._full_verify_warmup_steps
            and self._plan_count > 0
            and self._plan_count % self._full_verify_interval == 0
        )

    def _should_force_full_verify(self, *, pre_draft_step: int) -> bool:
        if pre_draft_step != self.mtp_step:
            return False
        if self._full_verify_update_count < self._full_verify_warmup_steps:
            return True
        if self._full_probe_pending:
            self._full_probe_pending = False
            return True
        return False

    def _record_plan_stats(
        self,
        *,
        req_num: int,
        dynamic_batch_size: int,
        draft_step: int,
        expected_tokens_per_req: float,
    ) -> None:
        self._plan_count += 1
        if self._plan_log_interval <= 0:
            return
        self._draft_step_counts[int(draft_step)] += 1
        self._verify_rows_per_req_sum += dynamic_batch_size / req_num
        self._expected_tokens_per_req_sum += expected_tokens_per_req
        if self._plan_count % self._plan_log_interval == 0:
            logger.info(
                "eagle3_dynamic_plan_stats plan_count=%d draft_step_counts=%s "
                "avg_verify_rows_per_req=%.6f avg_expected_tokens_per_req=%.6f "
                "static_tokens_per_req=%.6f full_verify_count=%d full_verify_req_count=%d "
                "observed_project_accept_ratio=%.6f observed_draft_accept_ratio=%.6f "
                "observed_tokens_per_req=%.6f target_verify_rows_per_req=%.6f "
                "prefix_accept_ratios=%s",
                self._plan_count,
                dict(sorted(self._draft_step_counts.items())),
                self._verify_rows_per_req_sum / self._plan_count,
                self._expected_tokens_per_req_sum / self._plan_count,
                self._full_verify_tokens_per_req_value,
                self._full_verify_update_count,
                self._full_verify_request_count,
                self._get_observed_project_accept_ratio(),
                self._get_observed_draft_accept_ratio(),
                self._get_observed_tokens_per_req(),
                self._get_controlled_verify_rows_per_req(),
                [round(value.get(), 6) for value in self.mtp_len_to_accept_ratio],
            )

    def _select_next_draft_step(self, *, req_num: int) -> int:
        """Choose a recoverable long-run Eagle3 draft length.

        For each possible length, jointly search the verify capacities that
        would be legal if that length were used repeatedly.  This is a
        one-state steady approximation to the next iteration and avoids
        crediting newly drafted tokens to the current target forward.
        """

        candidates = []
        min_expected_tokens = self._get_min_expected_tokens_per_req()
        for draft_step in range(self._max_dynamic_draft_step + 1):
            # Even a full verify at this proposal depth must be capable of
            # meeting the static-relative progress floor.  This prevents an
            # optimistic sparse (B, K) bucket from selecting draft_step=3 on
            # GSM8K and paying for many extra target iterations.
            max_depth_tokens_per_req = 1.0 + sum(
                self.mtp_len_to_accept_ratio[mtp_index].get() for mtp_index in range(draft_step)
            )
            if max_depth_tokens_per_req + 1e-6 < min_expected_tokens:
                continue
            max_batch_size = req_num * (draft_step + 1)
            dynamic_batch_size_keys = self._get_candidate_batch_sizes(
                req_num=req_num,
                max_batch_size=max_batch_size,
            )
            for dynamic_batch_size in dynamic_batch_size_keys:
                candidates.append(
                    self._get_eagle3_candidate(
                        req_num=req_num,
                        dynamic_batch_size=dynamic_batch_size,
                        verify_step=draft_step,
                        draft_step=draft_step,
                    )
                )
        if not candidates:
            return self._max_dynamic_draft_step
        return self._select_best_candidate(candidates, req_num=req_num)[4]

    def _get_eagle3_candidate(
        self,
        *,
        req_num: int,
        dynamic_batch_size: int,
        verify_step: int,
        draft_step: int,
    ) -> Tuple[float, int, float, float, int]:
        expected_token_num = self._estimate_expected_token_num(
            req_num=req_num,
            dynamic_batch_size=dynamic_batch_size,
            verify_step=verify_step,
        )
        cost_ms = self._get_eagle3_cost_ms(
            req_num=req_num,
            dynamic_batch_size=dynamic_batch_size,
            verify_step=verify_step,
            draft_step=draft_step,
            expected_token_num=expected_token_num,
        )
        expected_tokens_per_req = expected_token_num / req_num
        project_accept_ratio = max(0.0, min(1.0, expected_token_num / dynamic_batch_size))
        return (
            cost_ms,
            dynamic_batch_size,
            expected_tokens_per_req,
            project_accept_ratio,
            draft_step,
        )

    def _select_best_candidate(
        self,
        candidates: List[Tuple[float, int, float, float, int]],
        *,
        req_num: int,
    ) -> Tuple[float, int, float, float, int]:
        assert candidates
        min_expected_tokens = self._get_min_expected_tokens_per_req()
        min_verify_rows = self._get_min_verify_rows_per_req()
        progress_candidates = [
            candidate
            for candidate in candidates
            if candidate[2] >= min_expected_tokens and candidate[1] / req_num >= min_verify_rows
        ]
        efficient_candidates = [
            candidate for candidate in progress_candidates if candidate[3] >= self._min_project_accept_ratio
        ]
        if efficient_candidates:
            return min(efficient_candidates, key=lambda candidate: candidate[0])

        # If noisy online estimates leave no candidate satisfying both hard
        # constraints, minimize their worst relative violation.  This avoids
        # collapsing to the narrow acceptance-only plan or jumping to the
        # wide progress-only plan while the exact boundary bucket converges.
        drafted_candidates = [candidate for candidate in candidates if candidate[1] > req_num]
        if drafted_candidates:
            best_constraint_score = max(
                min(
                    candidate[2] / max(min_expected_tokens, 1e-6),
                    candidate[3] / max(self._min_project_accept_ratio, 1e-6),
                )
                for candidate in drafted_candidates
            )
            balanced_candidates = [
                candidate
                for candidate in drafted_candidates
                if min(
                    candidate[2] / max(min_expected_tokens, 1e-6),
                    candidate[3] / max(self._min_project_accept_ratio, 1e-6),
                )
                >= best_constraint_score - 1e-6
            ]
            return min(balanced_candidates, key=lambda candidate: candidate[0])

        # The current proposal may be too short to meet the floor after a
        # transition or drafting may be disabled.  Make maximal forward
        # progress instead of falling back to a deceptively cheap iteration.
        max_progress = max(candidate[2] for candidate in candidates)
        max_progress_candidates = [candidate for candidate in candidates if candidate[2] >= max_progress - 1e-6]
        return min(max_progress_candidates, key=lambda candidate: candidate[0])

    def _get_min_expected_tokens_per_req(self) -> float:
        if self._full_verify_request_count == 0:
            return 1.0
        target_tokens = max(
            1.0,
            self._full_verify_tokens_per_req_value * self._min_static_progress_ratio,
        )
        if self._observed_dynamic_request_count > 0 and self._get_observed_tokens_per_req() >= target_tokens:
            return max(1.0, target_tokens * self._progress_relax_ratio)
        return target_tokens

    def _get_min_verify_rows_per_req(self) -> float:
        min_expected_tokens = self._get_min_expected_tokens_per_req()
        if min_expected_tokens <= 1.0:
            return 1.0
        observed_project_accept_ratio = max(
            self._capacity_accept_ratio_floor,
            self._get_observed_project_accept_ratio(),
        )
        return min_expected_tokens / observed_project_accept_ratio

    def _estimate_expected_token_num(self, *, req_num: int, dynamic_batch_size: int, verify_step: int) -> float:
        accept_ratio = self._get_dynamic_batch_size_to_accept_ratio(
            req_num=req_num,
            dynamic_batch_size=dynamic_batch_size,
            verify_step=verify_step,
        )
        expected_token_num = min(
            dynamic_batch_size * accept_ratio,
            req_num * (verify_step + 1),
        )
        return max(float(req_num), expected_token_num)

    def _get_eagle3_cost_ms(
        self,
        *,
        req_num: int,
        dynamic_batch_size: int,
        verify_step: int,
        draft_step: int,
        expected_token_num: float = None,
    ) -> float:
        if expected_token_num is None:
            expected_token_num = self._estimate_expected_token_num(
                req_num=req_num,
                dynamic_batch_size=dynamic_batch_size,
                verify_step=verify_step,
            )

        # Eagle3 commit runs on all selected verify rows.  Every recurrent
        # proposal step after that runs on one accepted tail per request.
        draft_cost_ms = 0.0
        if draft_step > 0:
            draft_cost_ms = self.draft_model_speeds.get(dynamic_batch_size)
            if draft_step > 1:
                draft_cost_ms += self.draft_model_speeds.get(req_num) * (draft_step - 1)

        total_time_ms = self.main_model_speeds.get(dynamic_batch_size) + draft_cost_ms
        return total_time_ms / expected_token_num


class DSparkDynamicMTPPlanner(DynamicMTPPlanner):
    """DSpark confidence-scheduled verify-capacity planner."""

    planner_mode = "dspark"

    def __init__(self, mtp_step: int) -> None:
        super().__init__(
            mtp_step=mtp_step,
            use_random_mode=False,
        )
        self.mtp_len_to_accept_ratio = [
            _EMAValue(decay=0.95, init_value=1.0, enable_decay_warmup=True) for _ in range(self.mtp_step)
        ]
        self._predicted_dynamic_batch_sizes = deque(maxlen=2)

    def update_verified_prefix_stats(self, *, verify_len: int, accept_len: int) -> None:
        if verify_len - 1 <= 0:
            return
        max_mtp_len = min(verify_len - 1, self.mtp_step)
        for mtp_len in range(1, max_mtp_len + 1):
            self.update_mtp_len_to_accept_ratio(
                mtp_len=mtp_len,
                accept_ratio=1.0 if accept_len > mtp_len else 0.0,
            )
        return

    def update_predicted_schedule_probs(self, *, schedule_probs, req_num: int) -> None:
        """Record a confidence-derived future capacity estimate.

        The current decode step still routes rows by the current probabilities
        stored in req_to_next_token_probs.  This queue is only used to choose
        the future capacity K after a two-step delay, matching the asynchronous
        scheduler constraint described by DSpark.
        """

        if req_num <= 0:
            return
        if not self.main_model_speeds.has_data():
            return

        probs = self._to_numpy(schedule_probs)
        if probs is None or probs.ndim != 2 or probs.shape[1] <= 1:
            return

        draft_probs = probs[:, 1 : self.mtp_step + 1]
        if draft_probs.size == 0:
            return

        valid_rows = np.any(draft_probs > 0.0, axis=1)
        if not np.any(valid_rows):
            return

        conditional_probs = np.clip(draft_probs[valid_rows], 0.01, 0.99)
        survival_scores = np.cumprod(conditional_probs, axis=1)
        dynamic_batch_size = self._select_dynamic_batch_size_from_survival_scores(
            req_num=int(req_num),
            survival_scores=survival_scores,
        )
        self._predicted_dynamic_batch_sizes.append(dynamic_batch_size)
        return

    def get_dynamic_batch_size(self, req_num: int, original_batch_size: int) -> Tuple[int, int, int]:
        assert req_num * (self.mtp_step + 1) == original_batch_size
        pre_draft_step = self.pre_draft_step
        self.pre_draft_step = self.mtp_step
        if req_num == 0:
            return 0, self.mtp_step, pre_draft_step

        max_batch_size = req_num * (pre_draft_step + 1)
        if not self.main_model_speeds.has_data():
            return max_batch_size, self.mtp_step, pre_draft_step

        historical_batch_size = self._pop_historical_dynamic_batch_size(
            req_num=req_num,
            max_batch_size=max_batch_size,
        )
        if historical_batch_size is not None:
            return historical_batch_size, self.mtp_step, pre_draft_step
        if len(self._predicted_dynamic_batch_sizes) > 0:
            # A confidence estimate is available but has not satisfied the
            # two-step async delay yet.  Keep capacity conservative instead of
            # leaking a same-step EMA fallback into DSpark scheduling.
            return req_num, self.mtp_step, pre_draft_step

        candidate_batch_sizes = set(self.main_model_speeds.get_batch_size_keys_between(req_num, max_batch_size))
        candidate_batch_sizes.add(req_num)
        candidate_batch_sizes.add(max_batch_size)
        survival_prefix = self._estimate_survival_prefix(pre_draft_step)

        best_batch_size = req_num
        best_throughput = -float("inf")
        for dynamic_batch_size in sorted(candidate_batch_sizes):
            dynamic_batch_size = min(max(int(dynamic_batch_size), req_num), max_batch_size)
            expected_tokens = self._estimate_expected_tokens(
                req_num=req_num,
                dynamic_batch_size=dynamic_batch_size,
                survival_prefix=survival_prefix,
            )
            verify_ms = max(self.main_model_speeds.get(dynamic_batch_size), 1e-6)
            throughput = expected_tokens / verify_ms
            if throughput > best_throughput:
                best_throughput = throughput
                best_batch_size = dynamic_batch_size

        return best_batch_size, self.mtp_step, pre_draft_step

    def _pop_historical_dynamic_batch_size(self, *, req_num: int, max_batch_size: int) -> Optional[int]:
        if len(self._predicted_dynamic_batch_sizes) < 2:
            return None
        predicted_batch_size = int(self._predicted_dynamic_batch_sizes.popleft())
        return min(max(predicted_batch_size, int(req_num)), int(max_batch_size))

    def _select_dynamic_batch_size_from_survival_scores(
        self,
        *,
        req_num: int,
        survival_scores: np.ndarray,
    ) -> int:
        flat_survival_scores = survival_scores.reshape(-1)
        max_batch_size = int(req_num + flat_survival_scores.shape[0])
        if flat_survival_scores.shape[0] == 0:
            return int(req_num)

        candidate_batch_sizes = set(self.main_model_speeds.get_batch_size_keys_between(req_num, max_batch_size))
        candidate_batch_sizes.add(int(req_num))
        candidate_batch_sizes.add(max_batch_size)

        candidate_batch_sizes = {
            min(max(int(dynamic_batch_size), int(req_num)), max_batch_size)
            for dynamic_batch_size in candidate_batch_sizes
        }
        selected_draft_counts = sorted(
            {int(dynamic_batch_size) - int(req_num) for dynamic_batch_size in candidate_batch_sizes}
        )
        expected_accepts_by_count = self._topk_prefix_sums(
            values=flat_survival_scores,
            counts=selected_draft_counts,
        )

        best_batch_size = int(req_num)
        best_throughput = -float("inf")
        for dynamic_batch_size in sorted(candidate_batch_sizes):
            selected_draft_count = dynamic_batch_size - int(req_num)
            expected_tokens = float(req_num) + float(expected_accepts_by_count[selected_draft_count])
            verify_ms = max(self.main_model_speeds.get(dynamic_batch_size), 1e-6)
            throughput = expected_tokens / verify_ms
            if throughput > best_throughput:
                best_throughput = throughput
                best_batch_size = dynamic_batch_size
        return best_batch_size

    @staticmethod
    def _topk_prefix_sums(*, values: np.ndarray, counts: List[int]) -> Dict[int, float]:
        """Return sum(top-k(values)) only for the requested k values."""

        if len(counts) == 0:
            return {}

        flat_values = np.asarray(values, dtype=np.float64).reshape(-1)
        value_count = int(flat_values.shape[0])
        ans: Dict[int, float] = {}

        normalized_counts = sorted({min(max(int(count), 0), value_count) for count in counts})
        if value_count == 0:
            return {count: 0.0 for count in normalized_counts}

        partial_counts = [count for count in normalized_counts if 0 < count < value_count]
        if len(partial_counts) > 0 and partial_counts[-1] > value_count // 2:
            sorted_values = np.sort(flat_values)[::-1]
            prefix_sums = np.cumsum(sorted_values, dtype=np.float64)
            for count in normalized_counts:
                ans[count] = 0.0 if count == 0 else float(prefix_sums[count - 1])
            return ans

        if normalized_counts[0] == 0:
            ans[0] = 0.0
        if normalized_counts[-1] == value_count:
            ans[value_count] = float(np.sum(flat_values, dtype=np.float64))
        if len(partial_counts) == 0:
            return ans

        max_partial_count = partial_counts[-1]
        top_values = np.partition(flat_values, value_count - max_partial_count)[value_count - max_partial_count :]
        top_values.sort()
        top_values = top_values[::-1]
        prefix_sums = np.cumsum(top_values, dtype=np.float64)
        for count in partial_counts:
            ans[count] = float(prefix_sums[count - 1])
        return ans

    def _estimate_survival_prefix(self, pre_draft_step: int) -> List[float]:
        survival_prefix = [1.0]
        previous = 1.0
        for mtp_index in range(pre_draft_step):
            survival = float(self.mtp_len_to_accept_ratio[mtp_index].get())
            survival = max(0.0, min(previous, survival))
            survival_prefix.append(survival)
            previous = survival
        return survival_prefix

    @staticmethod
    def _estimate_expected_tokens(
        *,
        req_num: int,
        dynamic_batch_size: int,
        survival_prefix: List[float],
    ) -> float:
        expected_tokens = 0.0
        remaining = int(dynamic_batch_size)
        for survival in survival_prefix:
            take = min(req_num, remaining)
            if take <= 0:
                break
            expected_tokens += take * float(survival)
            remaining -= take
        return max(float(req_num), expected_tokens)

    @staticmethod
    def _to_numpy(value):
        if value is None:
            return None
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=np.float64)


class _InferCostMsTable:
    def __init__(self) -> None:
        self.infer_cost_ms_table = SortedDict()

    def update(self, batch_size: int, infer_cost_ms: float) -> None:
        assert batch_size > 0
        self.infer_cost_ms_table[int(batch_size)] = float(infer_cost_ms)
        return

    def has_data(self) -> bool:
        return len(self.infer_cost_ms_table) > 0

    def get(self, batch_size: int) -> float:
        assert batch_size > 0
        batch_size = int(batch_size)

        # 这种情况理论上不应该存在。
        if len(self.infer_cost_ms_table) == 0:
            return batch_size * 1000.0

        # 存在这个 batch_size 的记录，直接返回
        if batch_size in self.infer_cost_ms_table:
            return self.infer_cost_ms_table[batch_size]

        # 不存在这个 batch_size，需要进行插值估计。
        max_batch_size = self.infer_cost_ms_table.peekitem(-1)[0]
        if batch_size > max_batch_size:
            max_infer_cost_ms = self.infer_cost_ms_table.peekitem(-1)[1]
            # 这里面的 1000.0 意义是尽量使后续的估计，当超过最大graph支持的范围的时候，会直接倾向于关闭mtp功能。
            return max_infer_cost_ms + (batch_size - max_batch_size) * 1000.0
        else:
            # 找到第一个大于等于 batch_size 的 key，并返回它的 value。
            index = self.infer_cost_ms_table.bisect_left(batch_size)
            return self.infer_cost_ms_table.peekitem(index)[1]

    def get_batch_size_keys_between(self, batch_size1: int, batch_size2: int) -> List[int]:
        assert batch_size1 > 0 and batch_size2 > 0
        start = min(int(batch_size1), int(batch_size2))
        end = max(int(batch_size1), int(batch_size2))
        ans = list(self.infer_cost_ms_table.irange(minimum=start, maximum=end, inclusive=(True, True)))
        if len(ans) == 0:
            return [end]
        else:
            return ans

    def get_ceil_batch_size(self, batch_size: int, *, max_batch_size: int) -> Optional[int]:
        """Return the next recorded graph shape without inventing a key.

        The cost table can be sparse or empty when CUDA graph is disabled, so
        callers must be able to distinguish "no captured shape" from the
        requested upper bound.
        """

        if len(self.infer_cost_ms_table) == 0:
            return None
        index = self.infer_cost_ms_table.bisect_left(int(batch_size))
        if index >= len(self.infer_cost_ms_table):
            return None
        candidate = int(self.infer_cost_ms_table.peekitem(index)[0])
        if candidate > int(max_batch_size):
            return None
        return candidate


class _EMAValue:
    def __init__(self, decay: float, init_value: float, enable_decay_warmup: bool = True) -> None:
        assert decay > 0.0 and decay < 1.0
        self.enable_decay_warmup = enable_decay_warmup
        self.decay = decay

        if self.enable_decay_warmup:
            self.current_decay = 0.0
        else:
            self.current_decay = self.decay

        self.value = init_value
        self.second_moment_value = init_value ** 2
        self.update_count = 0

    def update(self, new_value: float):
        self.update_count += 1
        self.value = self.current_decay * self.value + (1.0 - self.current_decay) * new_value
        self.second_moment_value = self.current_decay * self.second_moment_value + (1.0 - self.current_decay) * (
            new_value ** 2
        )
        # 更新 current_decay 的值，使得 current_decay 逐渐逼近 decay 的值
        self.current_decay = min(self.decay, (self.decay + self.current_decay) / 2.0 + 0.001)
        return

    def get(self) -> float:
        return self.value

    def get_count(self) -> int:
        return self.update_count

    def get_second_moment(self) -> float:
        return self.second_moment_value

    def get_variance(self) -> float:
        return max(0.0, self.second_moment_value - self.value ** 2)

    def get_sigma(self) -> float:
        return math.sqrt(self.get_variance())


__all__ = [
    "DSparkDynamicMTPPlanner",
    "DynamicMTPPlanner",
    "Eagle3DynamicMTPPlanner",
    "FixedMTPPlanner",
    "SpecDecodePlan",
]
