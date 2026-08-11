from __future__ import annotations

import torch

from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
from lightllm.models.qwen3_dspark.model_output import DSparkModelOutput
from lightllm.server.router.model_infer.speculative.proposers.base import SpecProposal
from lightllm.server.router.model_infer.speculative.proposers.dflash import DFlashProposer


class DSparkProposer(DFlashProposer):
    """DSpark block proposer.

    DSpark shares DFlash's target-hidden KV injection and non-causal block
    backbone. Its post layer returns Markov-corrected logits and optional
    confidence logits, so the proposer follows the same token path as DFlash.
    """

    @torch.no_grad()
    def propose_next(
        self,
        main_model_input: ModelInput,
        main_model_output: ModelOutput,
        next_token_ids: torch.Tensor,
        b_req_mtp_start_loc: torch.Tensor,
        draft_step: int,
        accept_len: torch.Tensor | None = None,
    ) -> SpecProposal:
        assert 0 <= draft_step <= self.backend.max_draft_step
        assert accept_len is not None, "DSpark proposal requires target accept lengths"

        num_reqs = int(b_req_mtp_start_loc.shape[0])
        verify_row_count = next_token_ids.shape[0]
        draft_model = self.backend.draft_models[0]
        block_size = int(draft_model.block_size)
        assert block_size == self.backend.max_draft_step, (
            f"DSpark requires --mtp_step={block_size} for this checkpoint, " f"got {self.backend.max_draft_step}"
        )
        assert accept_len.shape[0] == num_reqs

        proposal_token_ids = next_token_ids.new_full(
            (verify_row_count, draft_step + 1),
            fill_value=1,
        )
        proposal_token_ids[:, 0] = next_token_ids
        schedule_probs = [] if self.enable_dynamic_spec else None

        assert main_model_output is not None and main_model_output.spec_hidden is not None
        self.extend_draft_kv_cache(
            main_model_input=main_model_input,
            target_hidden=main_model_output.spec_hidden,
        )

        if draft_step == 0:
            return SpecProposal(
                token_ids=proposal_token_ids,
                extra_mem_indexes_cpu=None,
                schedule_probs=schedule_probs,
            )

        selected_rows = self.select_accepted_tail_rows(
            b_req_mtp_start_loc=b_req_mtp_start_loc,
            accept_len=accept_len,
        )
        draft_input, draft_mem_indexes_cpu = self.build_block_draft_input(
            main_model_input=main_model_input,
            next_token_ids=next_token_ids,
            selected_rows=selected_rows,
            num_reqs=num_reqs,
        )
        draft_model_output = draft_model.forward(draft_input)
        assert isinstance(draft_model_output, DSparkModelOutput)

        expected_block_rows = num_reqs * block_size
        assert draft_model_output.logits.ndim >= 2, "draft logits must have a leading block-row dimension"
        assert (
            draft_model_output.logits.shape[0] == expected_block_rows
        ), f"draft logits rows must be {expected_block_rows}, got {draft_model_output.logits.shape[0]}"
        if draft_model_output.draft_token_ids is None:
            flat_token_ids = self.backend._gen_argmax_token_ids(draft_model_output)
        else:
            flat_token_ids = draft_model_output.draft_token_ids
        assert (
            flat_token_ids.numel() == expected_block_rows
        ), f"draft token rows must be {expected_block_rows}, got {flat_token_ids.numel()}"
        block_token_ids = flat_token_ids.reshape(num_reqs, block_size)
        proposal_token_ids[selected_rows, 1:] = block_token_ids[:, :draft_step]

        if self.enable_dynamic_spec:
            confidence_logits = draft_model_output.confidence_logits
            if confidence_logits is None:
                raise RuntimeError("DSpark dynamic verify requires confidence head logits")
            assert confidence_logits.ndim == 2, "confidence logits must be [selected_rows, block_size]"
            assert (
                confidence_logits.shape[0] == num_reqs
            ), f"confidence logits rows must be {num_reqs}, got {confidence_logits.shape[0]}"
            assert (
                confidence_logits.shape[1] == block_size
            ), f"confidence logits must have {block_size} columns, got {confidence_logits.shape[1]}"
            # Match the clamp used by the GPU dynamic row selector before it
            # converts conditional confidence to prefix survival probability.
            schedule_probs = self.scatter_selected_step_probs(
                selected_rows=selected_rows,
                selected_probs=confidence_logits[:, :draft_step].sigmoid().clamp(min=0.01, max=0.99),
                verify_row_count=verify_row_count,
            )

        return SpecProposal(
            token_ids=proposal_token_ids,
            extra_mem_indexes_cpu=draft_mem_indexes_cpu,
            schedule_probs=schedule_probs,
        )
