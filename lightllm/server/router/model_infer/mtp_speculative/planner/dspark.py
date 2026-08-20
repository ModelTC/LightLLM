from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

from lightllm.server.router.model_infer.mtp_speculative.planner.base import (
    BaseMtpPlanner,
    SpecDecodePlan,
    _InferCostMsTable,
)
from lightllm.server.router.model_infer.mtp_speculative.proposers.proposal_type import DSparkSpecProposal

if TYPE_CHECKING:
    from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend


class DSparkPlanner(BaseMtpPlanner):
    """DSpark's confidence-based verify-capacity planner.

    DSpark always drafts a complete block. Confidence from one proposal selects
    the target verify capacity two iterations later.
    """

    def __init__(self, backend: ModeBackend) -> None:
        self.backend = backend
        self.max_draft_step = int(backend.max_draft_step)
        self.block_size = int(backend.draft_models[0].block_size)
        self.target_infer_costs = _InferCostMsTable()
        self.draft_infer_costs = _InferCostMsTable()
        self._register_cuda_graph_costs()
        self._pending_verify_batch_sizes = deque(maxlen=2)

    def plan(self, decode_reqs: List, origin_batch_size: int) -> SpecDecodePlan:
        req_num = len(decode_reqs)
        full_batch_size = origin_batch_size
        dynamic_batch_size = full_batch_size
        delayed_batch_size = self._pop_delayed_batch_size(
            req_num=req_num,
            max_batch_size=full_batch_size,
        )
        if delayed_batch_size is not None:
            dynamic_batch_size = delayed_batch_size

        return SpecDecodePlan(
            origin_batch_size=origin_batch_size,
            dynamic_batch_size=dynamic_batch_size,
            draft_step=self.max_draft_step,
            pre_draft_step=self.max_draft_step,
        )

    def update_statics(
        self,
        plan: SpecDecodePlan,
        proposal: DSparkSpecProposal,
        req_num: int,
        accept_lengths,
    ) -> None:
        if proposal.schedule_scores_cpu is not None:
            self._update_confidence_probs(
                confidence_probs=proposal.schedule_scores_cpu,
                req_num=req_num,
            )

    def _get_draft_cost_ms(self, req_num: int, verify_batch_size: int, draft_step: int) -> float:
        """Return the cost of committing verify rows and generating one complete block."""

        extend_cost_ms = self.draft_infer_costs.estimate(verify_batch_size)
        block_cost_ms = self.draft_infer_costs.estimate(req_num * self.block_size)
        return extend_cost_ms + block_cost_ms

    def _register_cuda_graph_costs(self) -> None:
        target_graph = self.backend.model.graph
        if target_graph is not None:
            for batch_size, infer_cost_ms in target_graph.infer_cost_ms_by_batch_size.items():
                self.target_infer_costs.update(batch_size=batch_size, infer_cost_ms=infer_cost_ms)

        for draft_model in self.backend.draft_models:
            draft_graph = draft_model.graph
            if draft_graph is None:
                continue
            for batch_size, infer_cost_ms in draft_graph.infer_cost_ms_by_batch_size.items():
                self.draft_infer_costs.update(batch_size=batch_size, infer_cost_ms=infer_cost_ms)

    def _update_confidence_probs(self, confidence_probs, req_num: int) -> None:
        """Record a confidence-derived future capacity estimate.

        The current decode step still routes rows by the current probabilities
        stored in req_to_next_token_scores. This queue is only used to choose
        the future capacity K after a two-step delay, matching the asynchronous
        scheduler constraint described by DSpark.
        """

        if req_num <= 0:
            return

        probs = np.asarray(confidence_probs, dtype=np.float64)
        if probs.ndim != 2 or probs.shape[1] == 0:
            return

        draft_confidence_probs = probs[:, : self.max_draft_step]
        if draft_confidence_probs.size == 0:
            return

        # Proposal scores are dense request rows and contain draft columns only.
        conditional_probs = np.clip(draft_confidence_probs, 0.01, 0.99)
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
            round_ms = self.target_infer_costs.estimate(dynamic_batch_size) + self._get_draft_cost_ms(
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
