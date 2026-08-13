from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

import torch

from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput

if TYPE_CHECKING:
    from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend


@dataclass
class SpecProposal:
    """Candidate tokens and scheduling metadata produced by a proposer.

    `token_ids` has shape `[verify_batch, draft_step + 1]`; column 0 contains
    target-model tokens and the remaining columns contain draft candidates.
    `schedule_scores`, when present, has shape `[verify_batch, draft_step]`.
    Each column contains the proposer-specific score used by dynamic scheduling:
    selected-token probability for standard proposers, or confidence-head
    probability for DSpark.
    `schedule_scores_cpu` is the asynchronous CPU copy consumed by planners
    that use proposal scores directly.
    `extra_mem_indexes_cpu` tracks temporary KV slots owned by the proposal.
    """

    token_ids: torch.Tensor
    extra_mem_indexes_cpu: Optional[torch.Tensor]
    schedule_scores: Optional[torch.Tensor] = None
    schedule_scores_cpu: Optional[torch.Tensor] = None


class BaseSpecProposer:
    """Base class for algorithm-specific draft proposal generation.

    A proposer owns the draft-side state transition. The target model gives it
    the current target token ids plus captured target hidden features through
    the prefill-state and proposal hooks. The proposer returns candidate ids
    but does not verify acceptance; verification is handled by SpecEngine.
    """

    def __init__(self, *, backend: "ModeBackend", enable_dynamic_spec: bool) -> None:
        self.backend = backend
        self.enable_dynamic_spec = bool(enable_dynamic_spec)

    def get_draft_steps(self) -> Tuple[int, ...]:
        """Return the draft configurations supported by this proposer."""

        raise NotImplementedError

    def get_draft_cost_ms(
        self,
        draft_infer_costs,
        req_num: int,
        verify_batch_size: int,
        draft_step: int,
    ) -> float:
        """Return the complete draft cost for one ``(N, B, d)`` configuration."""

        raise NotImplementedError

    def select_accepted_tail_rows(self, b_req_mtp_start_loc: torch.Tensor, accept_len: torch.Tensor) -> torch.Tensor:
        return (b_req_mtp_start_loc + accept_len - 1).to(torch.long)

    def scatter_selected_step_probs(
        self,
        selected_rows: torch.Tensor,
        selected_probs: torch.Tensor,
        verify_row_count: int,
    ) -> torch.Tensor:
        out = torch.zeros(
            (verify_row_count, *selected_probs.shape[1:]),
            dtype=torch.float32,
            device=selected_probs.device,
        )
        out[selected_rows] = selected_probs.float()
        return out

    def alloc_extra_mem_indexes(self, token_count: int) -> torch.Tensor:
        """Allocate draft-owned temporary KV slots."""

        token_count = int(token_count)
        if token_count == 0:
            return torch.empty((0,), dtype=torch.int32, device="cpu")

        from lightllm.server.router.model_infer.infer_batch import g_infer_context

        if g_infer_context.radix_cache is not None:
            g_infer_context.radix_cache.free_radix_cache_to_get_enough_token(token_count)
        return g_infer_context.req_manager.mem_manager.alloc(token_count)

    def build_draft_state_from_prefill(
        self,
        target_model_input: ModelInput,
        target_model_output: ModelOutput,
        next_token_ids: torch.Tensor,
    ) -> None:
        """Build draft KV/state from target prefill before the first decode verify.

        Inputs:
        - `target_model_input`: target prompt ModelInput. Its request order and
          mem_indexes are reused by the draft state builder.
        - `target_model_output`: target output containing the features needed
          by the selected speculative algorithm.
        - `next_token_ids`: first accepted target token, shape [run_req_num].

        This hook only prepares draft-side state.  It intentionally does not
        scatter proposal tokens; the first decode iteration verifies as having
        no draft candidates and produces the first proposal through
        `propose_next`.
        """

        raise NotImplementedError

    def build_draft_state_from_prefill_overlap(
        self,
        target_model_input0: ModelInput,
        target_model_output0: ModelOutput,
        next_token_ids0: torch.Tensor,
        target_model_input1: ModelInput,
        target_model_output1: ModelOutput,
        next_token_ids1: torch.Tensor,
    ) -> None:
        """Build draft state from two overlapped target-prefill microbatches."""

        raise NotImplementedError

    def propose_next(
        self,
        main_model_input: ModelInput,
        main_model_output: ModelOutput,
        next_token_ids: torch.Tensor,
        b_req_mtp_start_loc: torch.Tensor,
        draft_step: int,
        accept_len: Optional[torch.Tensor] = None,
    ) -> SpecProposal:
        """Generate candidate tokens after one target decode forward.

        Inputs:
        - `main_model_input`: target decode ModelInput. In the fixed layout its
          batch is laid out as [req0-main, req0-draft1, ...]. Dynamic scheduling may
          compact this batch before target forward.
        - `next_token_ids`: target sampled ids for rows in `main_model_input`,
          shape [verify_batch].
        - `b_req_mtp_start_loc`: start row for each logical request inside the
          speculative verify batch, shape [logical_req_num].
        - `draft_step`: number of candidate draft tokens to produce.
        - `accept_len`: optional accepted-prefix length from the just-finished
          target forward. Stateful block proposers use it to commit the
          accepted target-hidden segment before preparing the next block.

        Returns a SpecProposal whose `token_ids[:, 0]` is `next_token_ids`.
        """

        raise NotImplementedError

    def propose_next_overlap(
        self,
        main_model_input0: ModelInput,
        main_model_output0: ModelOutput,
        next_token_ids0: torch.Tensor,
        real_verify_rows0: int,
        accept_len0: torch.Tensor,
        main_model_input1: ModelInput,
        main_model_output1: ModelOutput,
        next_token_ids1: torch.Tensor,
        real_verify_rows1: int,
        accept_len1: torch.Tensor,
        draft_step: int,
    ) -> SpecProposal:
        """Generate a proposal for two DP-overlapped microbatches."""

        raise NotImplementedError
