from __future__ import annotations

import torch

from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
from lightllm.server.router.model_infer.pin_mem_manager import g_pin_mem_manager
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
        num_reqs = int(b_req_mtp_start_loc.shape[0])
        verify_row_count = next_token_ids.shape[0]
        draft_model = self.backend.draft_models[0]
        block_size = int(draft_model.block_size)
        proposal_token_ids = next_token_ids.new_full(
            (verify_row_count, draft_step + 1),
            fill_value=1,
        )
        proposal_token_ids[:, 0] = next_token_ids
        schedule_scores = (
            torch.empty(
                (verify_row_count, draft_step),
                dtype=torch.float32,
                device=next_token_ids.device,
            )
            if self.enable_dynamic_spec
            else None
        )

        self.extend_draft_kv_cache(
            main_model_input=main_model_input,
            target_hidden=main_model_output.spec_hidden,
        )

        if draft_step == 0:
            return SpecProposal(
                token_ids=proposal_token_ids,
                extra_mem_indexes_cpu=None,
                schedule_scores=schedule_scores,
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

        if draft_model_output.draft_token_ids is None:
            flat_token_ids = self.backend._gen_argmax_token_ids(draft_model_output)
        else:
            flat_token_ids = draft_model_output.draft_token_ids
        block_token_ids = flat_token_ids.reshape(num_reqs, block_size)
        proposal_token_ids[selected_rows, 1:] = block_token_ids[:, :draft_step]

        if self.enable_dynamic_spec:
            confidence_logits = draft_model_output.confidence_logits
            if confidence_logits is None:
                raise RuntimeError("DSpark dynamic verify requires confidence head logits")
            # Match the clamp used by the GPU dynamic row selector before it
            # converts conditional confidence to prefix survival probability.
            schedule_scores = self.scatter_selected_step_probs(
                selected_rows=selected_rows,
                selected_probs=confidence_logits[:, :draft_step].sigmoid().clamp(min=0.01, max=0.99),
                verify_row_count=verify_row_count,
            )

        schedule_scores_cpu = None
        if schedule_scores is not None:
            schedule_scores_cpu = g_pin_mem_manager.async_copy_from_gpu_tensor(
                key="dspark_confidence_probs",
                gpu_tensor=schedule_scores,
            )

        return SpecProposal(
            token_ids=proposal_token_ids,
            extra_mem_indexes_cpu=draft_mem_indexes_cpu,
            schedule_scores=schedule_scores,
            schedule_scores_cpu=schedule_scores_cpu,
        )
