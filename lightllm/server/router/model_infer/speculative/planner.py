from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from sortedcontainers import SortedDict


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


__all__ = ["DSparkDynamicMTPPlanner", "DynamicMTPPlanner", "FixedMTPPlanner", "SpecDecodePlan"]
