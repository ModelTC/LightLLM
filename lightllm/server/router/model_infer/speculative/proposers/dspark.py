from __future__ import annotations

import torch

from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
from lightllm.server.router.model_infer.speculative.proposers.base import SpecProposal
from lightllm.server.router.model_infer.speculative.proposers.dflash import DFlashProposer


class DSparkProposer(DFlashProposer):
    """DSpark block proposer.

    DSpark shares DFlash's target-hidden KV injection and non-causal block
    backbone. Its post layer returns Markov-corrected logits and optional
    confidence logits, so the proposer follows the same token path as DFlash.
    """

    variant = "dspark"

    @torch.no_grad()
    def propose_next(
        self,
        *,
        main_model_input: ModelInput,
        main_model_output: ModelOutput = None,
        next_token_ids: torch.Tensor,
        b_req_mtp_start_loc: torch.Tensor,
        draft_step: int,
        verify_result=None,
    ) -> SpecProposal:
        del main_model_output
        assert 0 <= draft_step <= self.backend.mtp_step
        assert verify_result is not None, "DSpark proposal requires target verify result"

        num_reqs = int(b_req_mtp_start_loc.shape[0])
        verify_row_count = next_token_ids.shape[0]
        draft_model = self.backend.draft_models[0]
        block_size = int(draft_model.block_size)
        assert block_size >= draft_step
        assert verify_result.accept_len.shape[0] == num_reqs

        proposal_token_ids = next_token_ids.new_full(
            (verify_row_count, draft_step + 1),
            fill_value=1,
        )
        proposal_token_ids[:, 0] = next_token_ids
        schedule_probs = [] if self.enable_dynamic_mtp else None

        if draft_step == 0:
            return SpecProposal(
                token_ids=proposal_token_ids,
                extra_mem_indexes_cpu=None,
                draft_probs=[] if self.enable_dynamic_mtp else None,
                schedule_probs=schedule_probs,
            )

        self.extend_draft_kv_cache(main_model_input=main_model_input)
        selected_rows = self.select_accepted_tail_rows(
            b_req_mtp_start_loc=b_req_mtp_start_loc,
            accept_len=verify_result.accept_len,
        )
        draft_input, draft_mem_indexes_cpu = self.build_block_draft_input(
            main_model_input=main_model_input,
            next_token_ids=next_token_ids,
            selected_rows=selected_rows,
            num_reqs=num_reqs,
        )
        draft_model_output = draft_model.forward(draft_input)

        expected_block_rows = num_reqs * block_size
        assert draft_model_output.logits.ndim >= 2, "draft logits must have a leading block-row dimension"
        assert (
            draft_model_output.logits.shape[0] == expected_block_rows
        ), f"draft logits rows must be {expected_block_rows}, got {draft_model_output.logits.shape[0]}"
        flat_token_ids = self.backend._gen_argmax_token_ids(draft_model_output)
        assert (
            flat_token_ids.numel() == expected_block_rows
        ), f"draft token rows must be {expected_block_rows}, got {flat_token_ids.numel()}"
        block_token_ids = flat_token_ids.reshape(num_reqs, block_size)
        proposal_token_ids[selected_rows, 1:] = block_token_ids[:, :draft_step]

        draft_probs = None
        if self.enable_dynamic_mtp:
            confidence_logits = draft_model_output.mtp_draft_confidence_logits
            if confidence_logits is None:
                raise RuntimeError("DSpark dynamic verify requires confidence head logits")
            assert confidence_logits.ndim == 2, "confidence logits must be [selected_rows, block_size]"
            assert (
                confidence_logits.shape[0] == num_reqs
            ), f"confidence logits rows must be {num_reqs}, got {confidence_logits.shape[0]}"
            assert (
                confidence_logits.shape[1] >= draft_step
            ), f"confidence logits columns must cover draft_step={draft_step}, got {confidence_logits.shape[1]}"
            schedule_probs = self._scatter_step_probs(
                selected_rows=selected_rows,
                probs=confidence_logits[:, :draft_step].sigmoid(),
                verify_row_count=verify_row_count,
            )

        return SpecProposal(
            token_ids=proposal_token_ids,
            extra_mem_indexes_cpu=draft_mem_indexes_cpu,
            draft_probs=draft_probs,
            schedule_probs=schedule_probs,
        )

    def _scatter_step_probs(
        self,
        *,
        selected_rows: torch.Tensor,
        probs: torch.Tensor,
        verify_row_count: int,
    ):
        assert selected_rows.ndim == 1, "selected_rows must be 1D"
        assert probs.ndim == 2, "confidence probabilities must be [selected_rows, draft_step]"
        assert probs.shape[0] == selected_rows.shape[0], (
            "confidence probability rows must match selected rows: " f"{selected_rows.shape[0]}, got {probs.shape[0]}"
        )
        # Keep the async CPU capacity estimate aligned with the GPU dynamic
        # selector, which clamps conditional draft probabilities before
        # converting them to prefix survival scores.  Unselected rows remain
        # zero because scatter_selected_step_probs initializes the output.
        probs = probs.clamp(min=0.01, max=0.99)
        out = probs.new_zeros((verify_row_count, probs.shape[1]), dtype=torch.float32)
        out[selected_rows, :] = probs.float()
        return out
