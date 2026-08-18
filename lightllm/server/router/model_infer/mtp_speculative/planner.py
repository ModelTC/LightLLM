from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
from sortedcontainers import SortedDict

if TYPE_CHECKING:
    from lightllm.server.router.model_infer.mtp_speculative.proposers.base import BaseSpecProposer


@dataclass(frozen=True)
class SpecDecodePlan:
    """Planner decision for one target decode iteration.

    Fixed scheduling uses the full speculative-expanded target batch:
    - dynamic_batch_size is None
    - draft_step == max_draft_step

    Dynamic speculative scheduling may compact target rows before forward:
    - dynamic_batch_size is the selected target row count
    - draft_step is the candidate length to generate after target verify
    - pre_draft_step describes the previous iteration and controls whether
      GPU verify sync can be skipped
    """

    dynamic_batch_size: Optional[int]
    draft_step: int
    pre_draft_step: int
    # False when the current batch contains requests without a proposal from
    # the previous iteration. Such an iteration does not represent one
    # well-defined LightSpec runtime configuration.
    all_reqs_have_proposals: bool = True

    @property
    def is_dynamic(self) -> bool:
        return self.dynamic_batch_size is not None

    @property
    def skip_verify_sync(self) -> bool:
        return self.is_dynamic and self.pre_draft_step == 0

    def filter_reqs(self, reqs: List, selected_row_mask_cpu) -> List:
        return [req for req, selected in zip(reqs, selected_row_mask_cpu.tolist()) if selected]


class FixedSpecPlanner:
    """Planner for fixed-width speculative decoding."""

    def __init__(self, max_draft_step: int) -> None:
        self.max_draft_step = int(max_draft_step)

    def plan(self, req_num: int | None = None, original_batch_size: int | None = None) -> SpecDecodePlan:
        return SpecDecodePlan(
            dynamic_batch_size=None,
            draft_step=self.max_draft_step,
            pre_draft_step=self.max_draft_step,
        )


class LightSpecPlanner:
    """Choose the current verify budget and the next draft configuration.

    LightSpec evaluates a runtime configuration as ``(N, B, d)``: logical
    requests, physical target rows, and draft configuration. Planning follows
    the paper's budget-then-fill order. It chooses the current ``B`` within the
    proposal produced by ``pre_draft_step``. The next ``draft_step`` is selected
    from self-consistent ``(B, d)`` configurations for the following iteration,
    so a short current proposal cannot prevent the planner from drafting deeper
    again. Candidate identities belong to the GPU Fill stage.

    Each configuration minimizes estimated milliseconds per committed token:
    ``(target_cost(B) + draft_cost(N, B, d)) / expected_progress(N, B, d)``.
    Progress is learned from one normalized batch observation ``sum(accept_len)
    / B`` per iteration, then smoothed with an EMA. It is never updated once per
    request; doing so would make adaptation depend on concurrency.

    The proposer supplies its valid draft configurations and complete draft
    cost. The planner therefore stays independent of the proposal algorithm and
    its physical execution layout.
    """

    def __init__(
        self,
        max_draft_step: int,
        draft_cost_provider: "BaseSpecProposer",
    ) -> None:
        self.max_draft_step = int(max_draft_step)
        self.draft_cost_provider = draft_cost_provider
        self.draft_steps = draft_cost_provider.get_draft_steps()

        self.target_infer_costs = _InferCostMsTable()
        self.draft_infer_costs = _InferCostMsTable()

        # Each observation is the normalized committed progress U / B from one
        # complete batch. The draft configuration is part of the key because a
        # deeper candidate pool may improve Fill even at the same (N, B).
        self.progress_ema_by_config: Dict[Tuple[int, int, int], _EMAValue] = {}
        # Full-width verification provides an unbiased survival probability
        # for every draft depth. It is the fallback for configurations that
        # have not produced their own progress observation yet.
        self.prefix_survival_by_depth: List[Optional[_EMAValue]] = [None] * self.max_draft_step

        # The current verify width is bounded by the proposal built last time.
        self.pre_draft_step = self.max_draft_step

    def plan(self, req_num: int, original_batch_size: int, proposal_req_num: int) -> SpecDecodePlan:
        pre_draft_step = self.pre_draft_step
        if req_num == 0:
            self.pre_draft_step = self.max_draft_step
            return SpecDecodePlan(
                dynamic_batch_size=0,
                draft_step=self.max_draft_step,
                pre_draft_step=pre_draft_step,
            )

        # A request entering its first decode has only the guaranteed target
        # row. Existing requests additionally expose pre_draft_step candidates.
        # This bounds Verify by the candidate pool that physically exists.
        available_batch_size = req_num + proposal_req_num * pre_draft_step
        max_batch_size = min(original_batch_size, available_batch_size)
        all_reqs_have_proposals = proposal_req_num == req_num

        if (
            not self.target_infer_costs.has_data()
            or not self.draft_infer_costs.has_data()
            or not self.progress_ema_by_config
        ):
            # Costs come from CUDA Graph capture, while progress requires one
            # real verification batch. Keep the available proposal intact until
            # both parts of J(N, B, d) have an observation.
            self.pre_draft_step = self.max_draft_step
            return SpecDecodePlan(
                dynamic_batch_size=max_batch_size,
                draft_step=self.max_draft_step,
                pre_draft_step=pre_draft_step,
                all_reqs_have_proposals=all_reqs_have_proposals,
            )

        min_batch_size = req_num
        batch_sizes = self.target_infer_costs.get_batch_size_keys_between(min_batch_size, max_batch_size)
        costs = [
            self._get_cost_ms(
                req_num=req_num,
                dynamic_batch_size=dynamic_batch_size,
                draft_step=pre_draft_step,
            )
            for dynamic_batch_size in batch_sizes
        ]
        dynamic_batch_size = batch_sizes[np.argmin(costs)]
        draft_step = self._select_draft_step(req_num=req_num)

        self.pre_draft_step = draft_step
        return SpecDecodePlan(
            dynamic_batch_size=dynamic_batch_size,
            draft_step=draft_step,
            pre_draft_step=pre_draft_step,
            all_reqs_have_proposals=all_reqs_have_proposals,
        )

    def update_infer_cost(self, batch_size: int, infer_cost_ms: float, is_draft_model: bool) -> None:
        cost_table = self.draft_infer_costs if is_draft_model else self.target_infer_costs
        cost_table.update(batch_size=batch_size, infer_cost_ms=infer_cost_ms)

    def update_feedback(
        self,
        plan: SpecDecodePlan,
        req_num: int,
        accept_lengths,
        schedule_scores=None,
    ) -> None:
        # The progress EMA records one complete-batch sample for a single
        # (N, B, d) configuration. all_reqs_have_proposals is false if any request is
        # on its first decode and therefore has no preceding proposal; its
        # structural accept_len=1 would otherwise bias that configuration's
        # progress downward. Skip the mixed batch instead of partially
        # updating the batch-level statistic with different N/B semantics.
        if not plan.all_reqs_have_proposals:
            return
        self.update_verified_batch(
            accept_lengths=accept_lengths,
            req_num=req_num,
            dynamic_batch_size=plan.dynamic_batch_size,
            verified_draft_step=plan.pre_draft_step,
        )

    def update_verified_batch(
        self,
        accept_lengths,
        req_num: int,
        dynamic_batch_size: int,
        verified_draft_step: int,
    ) -> None:
        """Record one batch-level progress sample for the verified configuration."""

        accept_lengths = np.asarray(accept_lengths)
        if accept_lengths.size == 0:
            return

        config = (int(req_num), int(dynamic_batch_size), int(verified_draft_step))
        progress = float(accept_lengths.sum()) / dynamic_batch_size
        if config not in self.progress_ema_by_config:
            self.progress_ema_by_config[config] = _EMAValue(
                decay=0.9,
                init_value=progress,
            )
        self.progress_ema_by_config[config].update(progress)

        if dynamic_batch_size == req_num * (verified_draft_step + 1):
            for depth in range(1, verified_draft_step + 1):
                survival = float(np.mean(accept_lengths > depth))
                survival_ema = self.prefix_survival_by_depth[depth - 1]
                if survival_ema is None:
                    survival_ema = _EMAValue(decay=0.9, init_value=survival)
                    self.prefix_survival_by_depth[depth - 1] = survival_ema
                survival_ema.update(survival)

    def _select_draft_step(self, req_num: int) -> int:
        """Choose the best self-consistent next proposal depth.

        The current target batch is bounded by ``pre_draft_step``, but the
        proposal generated now is consumed by the next iteration. Evaluate
        each candidate depth with the verify widths that depth can create so a
        short current proposal cannot permanently prevent deeper drafting.
        """

        best_cost_ms = float("inf")
        best_draft_step = self.draft_steps[0]
        for draft_step in self.draft_steps:
            max_batch_size = req_num * (draft_step + 1)
            batch_sizes = self.target_infer_costs.get_batch_size_keys_between(req_num, max_batch_size)
            for dynamic_batch_size in batch_sizes:
                cost_ms = self._get_cost_ms(
                    req_num=req_num,
                    dynamic_batch_size=dynamic_batch_size,
                    draft_step=draft_step,
                )
                if cost_ms < best_cost_ms:
                    best_cost_ms = cost_ms
                    best_draft_step = draft_step
        return best_draft_step

    def _get_cost_ms(self, req_num: int, dynamic_batch_size: int, draft_step: int) -> float:
        """Estimate milliseconds per committed token for one configuration."""

        accept_ratio = self._estimate_progress(
            req_num=req_num,
            dynamic_batch_size=dynamic_batch_size,
            draft_step=draft_step,
        )
        total_time = self.target_infer_costs.get(dynamic_batch_size) + self.draft_cost_provider.get_draft_cost_ms(
            draft_infer_costs=self.draft_infer_costs,
            req_num=req_num,
            verify_batch_size=dynamic_batch_size,
            draft_step=draft_step,
        )
        token_num = min((dynamic_batch_size * accept_ratio), req_num * (draft_step + 1))
        token_num = max(token_num, req_num)
        cost_ms = total_time / token_num
        return cost_ms

    def _estimate_progress(self, req_num: int, dynamic_batch_size: int, draft_step: int) -> float:
        config = (int(req_num), int(dynamic_batch_size), int(draft_step))
        ema = self.progress_ema_by_config.get(config)
        return ema.get() if ema is not None else self._estimate_prefix_progress(*config)

    def _estimate_prefix_progress(self, req_num: int, dynamic_batch_size: int, draft_step: int) -> float:
        if not any(self.prefix_survival_by_depth):
            return 1.0

        remaining_draft_rows = dynamic_batch_size - req_num
        expected_tokens = float(req_num)
        for depth in range(min(draft_step, self.max_draft_step)):
            selected_rows = min(req_num, remaining_draft_rows)
            if selected_rows <= 0:
                break
            survival_ema = self.prefix_survival_by_depth[depth]
            survival = 1.0 if survival_ema is None else survival_ema.get()
            expected_tokens += selected_rows * survival
            remaining_draft_rows -= selected_rows
        return expected_tokens / dynamic_batch_size


class DSparkPlanner:
    """DSpark's confidence-based verify-capacity planner.

    DSpark always drafts a complete block. Confidence from one proposal selects
    the target verify capacity two iterations later.
    """

    def __init__(self, max_draft_step: int, draft_cost_provider: "BaseSpecProposer") -> None:
        self.max_draft_step = int(max_draft_step)
        self.draft_cost_provider = draft_cost_provider
        self.target_infer_costs = _InferCostMsTable()
        self.draft_infer_costs = _InferCostMsTable()
        self._pending_verify_batch_sizes = deque(maxlen=2)

    def plan(self, req_num: int, original_batch_size: int) -> SpecDecodePlan:
        if req_num == 0:
            return SpecDecodePlan(
                dynamic_batch_size=0,
                draft_step=self.max_draft_step,
                pre_draft_step=self.max_draft_step,
            )

        full_batch_size = original_batch_size
        dynamic_batch_size = full_batch_size
        if self.target_infer_costs.has_data() and self.draft_infer_costs.has_data():
            delayed_batch_size = self._pop_delayed_batch_size(
                req_num=req_num,
                max_batch_size=full_batch_size,
            )
            if delayed_batch_size is not None:
                dynamic_batch_size = delayed_batch_size

        return SpecDecodePlan(
            dynamic_batch_size=dynamic_batch_size,
            draft_step=self.max_draft_step,
            pre_draft_step=self.max_draft_step,
        )

    def update_infer_cost(self, batch_size: int, infer_cost_ms: float, is_draft_model: bool) -> None:
        cost_table = self.draft_infer_costs if is_draft_model else self.target_infer_costs
        cost_table.update(batch_size=batch_size, infer_cost_ms=infer_cost_ms)

    def update_feedback(
        self,
        plan: SpecDecodePlan,
        req_num: int,
        accept_lengths,
        schedule_scores=None,
    ) -> None:
        if schedule_scores is not None:
            self.update_confidence_probs(
                confidence_probs=schedule_scores,
                req_num=req_num,
            )

    def update_confidence_probs(self, confidence_probs, req_num: int) -> None:
        """Record a confidence-derived future capacity estimate.

        The current decode step still routes rows by the current probabilities
        stored in req_to_next_token_scores. This queue is only used to choose
        the future capacity K after a two-step delay, matching the asynchronous
        scheduler constraint described by DSpark.
        """

        if req_num <= 0:
            return
        if not self.target_infer_costs.has_data() or not self.draft_infer_costs.has_data():
            return

        probs = np.asarray(confidence_probs, dtype=np.float64)
        if probs.ndim != 2 or probs.shape[1] == 0:
            return

        draft_confidence_probs = probs[:, : self.max_draft_step]
        if draft_confidence_probs.size == 0:
            return

        # Confidence is scattered onto one accepted-tail row per request;
        # unused verify rows remain zero.
        valid_rows = np.any(draft_confidence_probs > 0.0, axis=1)
        if not np.any(valid_rows):
            return

        conditional_probs = np.clip(draft_confidence_probs[valid_rows], 0.01, 0.99)
        survival_scores = np.cumprod(conditional_probs, axis=1)
        dynamic_batch_size = self._select_dynamic_batch_size_from_survival_scores(
            req_num=int(req_num),
            survival_scores=survival_scores,
        )
        self._pending_verify_batch_sizes.append(dynamic_batch_size)

    def _pop_delayed_batch_size(self, req_num: int, max_batch_size: int) -> Optional[int]:
        if len(self._pending_verify_batch_sizes) < 2:
            return None
        predicted_batch_size = int(self._pending_verify_batch_sizes.popleft())
        return min(max(predicted_batch_size, int(req_num)), int(max_batch_size))

    def _select_dynamic_batch_size_from_survival_scores(
        self,
        req_num: int,
        survival_scores: np.ndarray,
    ) -> int:
        flat_survival_scores = survival_scores.reshape(-1)
        max_batch_size = int(req_num + flat_survival_scores.shape[0])
        if flat_survival_scores.shape[0] == 0:
            return int(req_num)

        candidate_batch_sizes = set(self.target_infer_costs.get_batch_size_keys_between(req_num, max_batch_size))
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
            round_ms = self.target_infer_costs.get(dynamic_batch_size) + self.draft_cost_provider.get_draft_cost_ms(
                draft_infer_costs=self.draft_infer_costs,
                req_num=req_num,
                verify_batch_size=dynamic_batch_size,
                draft_step=self.max_draft_step,
            )
            throughput = expected_tokens / max(round_ms, 1e-6)
            if throughput > best_throughput:
                best_throughput = throughput
                best_batch_size = dynamic_batch_size
        return best_batch_size

    @staticmethod
    def _topk_prefix_sums(values: np.ndarray, counts: List[int]) -> Dict[int, float]:
        """Return sum(top-k(values)) only for the requested k values."""

        if not counts:
            return {}

        flat_values = np.asarray(values, dtype=np.float64).reshape(-1)
        value_count = int(flat_values.shape[0])
        normalized_counts = sorted({min(max(int(count), 0), value_count) for count in counts})
        if value_count == 0:
            return {count: 0.0 for count in normalized_counts}

        prefix_sums = np.concatenate(([0.0], np.cumsum(np.sort(flat_values)[::-1], dtype=np.float64)))
        return {count: float(prefix_sums[count]) for count in normalized_counts}


class _InferCostMsTable:
    def __init__(self) -> None:
        self.infer_cost_ms_table = SortedDict()

    def update(self, batch_size: int, infer_cost_ms: float) -> None:
        self.infer_cost_ms_table[int(batch_size)] = float(infer_cost_ms)

    def has_data(self) -> bool:
        return len(self.infer_cost_ms_table) > 0

    def get(self, batch_size: int) -> float:
        batch_size = int(batch_size)

        if len(self.infer_cost_ms_table) == 0:
            return batch_size * 1000.0

        if batch_size in self.infer_cost_ms_table:
            return self.infer_cost_ms_table[batch_size]

        max_batch_size = self.infer_cost_ms_table.peekitem(-1)[0]
        if batch_size > max_batch_size:
            max_infer_cost_ms = self.infer_cost_ms_table.peekitem(-1)[1]
            # 超过最大 graph 范围时使用高惩罚，使调度器倾向于关闭 speculative draft。
            return max_infer_cost_ms + (batch_size - max_batch_size) * 1000.0

        index = self.infer_cost_ms_table.bisect_left(batch_size)
        return self.infer_cost_ms_table.peekitem(index)[1]

    def estimate(self, batch_size: int) -> float:
        """Estimate an uncaptured batch without applying the graph-miss penalty."""

        batch_size = int(batch_size)
        max_batch_size, max_cost_ms = self.infer_cost_ms_table.peekitem(-1)
        if batch_size <= max_batch_size:
            return self.get(batch_size)
        return max_cost_ms * batch_size / max_batch_size

    def get_batch_size_keys_between(self, batch_size1: int, batch_size2: int) -> List[int]:
        start = min(int(batch_size1), int(batch_size2))
        end = max(int(batch_size1), int(batch_size2))
        batch_sizes = set(self.infer_cost_ms_table.irange(minimum=start, maximum=end, inclusive=(True, True)))
        batch_sizes.update((start, end))
        return sorted(batch_sizes)


class _EMAValue:
    def __init__(self, decay: float, init_value: float) -> None:
        self.decay = decay
        self.value = init_value
        self.update_count = 0

    def update(self, new_value: float):
        self.update_count += 1
        self.value = self.decay * self.value + (1.0 - self.decay) * new_value

    def get(self) -> float:
        return self.value

    def get_count(self) -> int:
        return self.update_count


__all__ = [
    "DSparkPlanner",
    "FixedSpecPlanner",
    "LightSpecPlanner",
    "SpecDecodePlan",
]
