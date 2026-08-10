from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Union

import torch

from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput

if TYPE_CHECKING:
    from lightllm.server.router.model_infer.speculative.engine import SpecEngine


@dataclass
class SpecProposal:
    """Draft proposal returned by a proposer.

    `token_ids` is the LightLLM service equivalent of DeepSpec's
    DraftProposal.verify_input_ids.  It contains the target model's freshly
    sampled token in column 0, followed by draft candidates:

        token_ids: [verify_batch, draft_step + 1]

    With fixed scheduling, `draft_step == backend.max_draft_step`. Dynamic speculative
    scheduling may make it shorter, and the engine pads before scatter.

    `draft_probs` is intentionally narrower than DeepSpec's full
    [B, K, vocab] probability tensor. The current dynamic scheduler only needs
    the selected-token probability from each draft step:

        draft_probs[i]: [verify_batch]

    `schedule_probs` optionally overrides `draft_probs` for dynamic verify
    row selection.  It can be a list of per-step vectors or a dense
    [verify_batch, draft_step] matrix.  DSpark uses confidence-head conditional
    acceptance probabilities here; the engine scatters them into the same
    per-request buffer and the dynamic selector converts them to prefix
    survival probabilities.

    `extra_mem_indexes_cpu` records draft-only KV slots allocated by recurrent
    or block proposers.  Eagle3 uses these slots for recurrent draft tokens;
    DFlash uses them for current-block scratch query/mask K/V:

        extra_mem_indexes_cpu: [slot_count]

    """

    token_ids: torch.Tensor
    extra_mem_indexes_cpu: Optional[torch.Tensor]
    draft_probs: Optional[List[torch.Tensor]] = None
    schedule_probs: Optional[Union[List[torch.Tensor], torch.Tensor]] = None


class BaseSpecProposer:
    """Base class for algorithm-specific draft proposal generation.

    A proposer owns the draft-side state transition.  The target model gives it
    the current target token ids plus captured target hidden features through
    SpecEngine.prepare_draft_* methods.  The proposer returns candidate ids
    but does not verify acceptance; verification is handled by SpecVerifier.
    """

    def __init__(self, engine: "SpecEngine") -> None:
        self.engine = engine
        self.backend = engine.backend

    @property
    def enable_dynamic_spec(self) -> bool:
        return self.engine.enable_dynamic_spec

    def prepare_draft_prefill_input(
        self,
        model_input: ModelInput,
        next_token_ids: torch.Tensor,
        mtp_draft_input_hiddens: torch.Tensor,
    ) -> ModelInput:
        return self.engine.prepare_draft_prefill_input(
            model_input=model_input,
            next_token_ids=next_token_ids,
            mtp_draft_input_hiddens=mtp_draft_input_hiddens,
        )

    def prepare_draft_decode_input(
        self,
        model_input: ModelInput,
        next_token_ids: torch.Tensor,
        mtp_draft_input_hiddens: torch.Tensor,
    ) -> ModelInput:
        return self.engine.prepare_draft_decode_input(
            model_input=model_input,
            next_token_ids=next_token_ids,
            mtp_draft_input_hiddens=mtp_draft_input_hiddens,
        )

    def select_accepted_tail_rows(self, b_req_mtp_start_loc: torch.Tensor, accept_len: torch.Tensor) -> torch.Tensor:
        return (b_req_mtp_start_loc + accept_len - 1).to(torch.long)

    def scatter_selected_step_probs(
        self,
        selected_rows: torch.Tensor,
        selected_probs: torch.Tensor,
        verify_row_count: int,
    ) -> torch.Tensor:
        out = torch.zeros(
            (verify_row_count,),
            dtype=torch.float32,
            device=selected_probs.device,
        )
        out[selected_rows] = selected_probs.float()
        return out

    def alloc_extra_mem_indexes(self, token_count: int) -> torch.Tensor:
        """Allocate draft-owned temporary KV slots."""

        return self.engine.alloc_extra_mem_indexes(token_count)

    def build_initial_draft_state(
        self,
        model_input: ModelInput,
        model_output: ModelOutput,
        next_token_ids: torch.Tensor,
    ) -> None:
        """Build initial draft KV/state before the first decode verify step.

        Inputs:
        - `model_input`: target prompt ModelInput.  Its request order and
          mem_indexes are reused by the draft state builder.
        - `next_token_ids`: first accepted target token, shape [run_req_num].

        SpecEngine.prepare_draft_prefill_input injects captured target hidden
        features into `mtp_draft_input_hiddens`.

        This hook only prepares draft-side state.  It intentionally does not
        scatter proposal tokens; the first decode iteration verifies as having
        no draft candidates and produces the first proposal through
        `propose_next`.
        """

        raise NotImplementedError

    def build_initial_draft_state_overlap(
        self,
        model_input0: ModelInput,
        model_output0: ModelOutput,
        next_token_ids0: torch.Tensor,
        model_input1: ModelInput,
        model_output1: ModelOutput,
        next_token_ids1: torch.Tensor,
    ) -> None:
        """Build initial draft state for two overlapped prefill microbatches."""

        raise NotImplementedError

    def propose_next(
        self,
        main_model_input: ModelInput,
        main_model_output: Optional[ModelOutput],
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
