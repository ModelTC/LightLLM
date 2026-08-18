from __future__ import annotations

import torch

from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
from lightllm.server.router.model_infer.mtp_speculative.proposers.base import SpecProposal
from lightllm.server.router.model_infer.mtp_speculative.proposers.parallel_block import ParallelBlockProposer


class DFlashProposer(ParallelBlockProposer):
    """DFlash block-diffusion proposer.

    The drafter predicts a complete token block in one parallel forward from
    the accepted-tail anchor and mask-token positions.
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
        request_count = int(b_req_mtp_start_loc.shape[0])
        verify_row_count = int(next_token_ids.shape[0])
        draft_model = self.backend.draft_models[0]
        block_size = int(draft_model.block_size)
        proposal_token_ids = next_token_ids.new_full(
            (verify_row_count, draft_step + 1),
            fill_value=1,
        )
        proposal_token_ids[:, 0] = next_token_ids

        # One accepted-tail anchor expands to a complete block-diffusion draft.
        accepted_tail_rows = (b_req_mtp_start_loc + accept_len - 1).long()
        draft_input, extra_mem_indexes_cpu = self.build_block_draft_input(
            main_model_input=main_model_input,
            next_token_ids=next_token_ids,
            accepted_tail_rows=accepted_tail_rows,
            request_count=request_count,
        )
        self.extend_draft_kv_cache(
            main_model_input=main_model_input,
            target_hidden=main_model_output.mtp_collector.spec_hidden,
        )
        draft_output = draft_model.forward(draft_input)

        if self.enable_dynmaic_mtp:
            flat_draft_token_ids, flat_draft_token_probs = self.backend._gen_argmax_token_ids_and_prob(draft_output)
        else:
            flat_draft_token_ids = self.backend._gen_argmax_token_ids(draft_output)
        block_draft_token_ids = flat_draft_token_ids.reshape(request_count, block_size)
        proposal_token_ids[accepted_tail_rows, 1:] = block_draft_token_ids[:, :draft_step]

        schedule_scores = None
        if self.enable_dynmaic_mtp:
            block_draft_token_probs = flat_draft_token_probs.reshape(request_count, block_size)
            schedule_scores = torch.zeros(
                (verify_row_count, draft_step),
                dtype=torch.float32,
                device=next_token_ids.device,
            )
            schedule_scores[accepted_tail_rows] = block_draft_token_probs[:, :draft_step].float()
        return SpecProposal(
            token_ids=proposal_token_ids,
            extra_mem_indexes_cpu=extra_mem_indexes_cpu,
            schedule_scores=schedule_scores,
        )
