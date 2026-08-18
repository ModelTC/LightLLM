from __future__ import annotations

import torch

from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
from lightllm.server.router.model_infer.mtp_speculative.proposers.base import BaseSpecProposer, SpecProposal


class VanillaMTPProposer(BaseSpecProposer):
    """Chained MTP proposer.

    Each draft depth uses an independent MTP module. Module i consumes the
    hidden state produced by module i - 1 and predicts the next candidate.
    """

    def get_draft_steps(self):
        return tuple(range(self.backend.max_draft_step + 1))

    def get_draft_cost_ms(
        self,
        draft_infer_costs,
        req_num: int,
        verify_batch_size: int,
        draft_step: int,
    ) -> float:
        # Every selected MTP module processes the complete verify batch.
        return draft_infer_costs.get(verify_batch_size) * draft_step

    def build_draft_state_from_prefill(
        self,
        target_model_input: ModelInput,
        target_model_output: ModelOutput,
        next_token_ids: torch.Tensor,
    ) -> None:
        from lightllm.server.router.model_infer.mode_backend.mtp_pre_process import prepare_mtp_prefill_inputs

        draft_hidden = target_model_output.mtp_collector.spec_hidden
        draft_token_ids = next_token_ids
        for draft_model in self.backend.draft_models:
            prepare_mtp_prefill_inputs(
                model_input=target_model_input,
                b_next_token_ids=draft_token_ids,
                mtp_draft_input_hiddens=draft_hidden,
            )
            draft_output = draft_model.forward(target_model_input)
            draft_hidden = draft_output.mtp_collector.spec_hidden
            draft_token_ids = self.backend._gen_argmax_token_ids(draft_output)

    def build_draft_state_from_prefill_overlap(
        self,
        target_model_input0: ModelInput,
        target_model_output0: ModelOutput,
        next_token_ids0: torch.Tensor,
        target_model_input1: ModelInput,
        target_model_output1: ModelOutput,
        next_token_ids1: torch.Tensor,
    ) -> None:
        from lightllm.server.router.model_infer.mode_backend.mtp_pre_process import prepare_mtp_prefill_inputs

        draft_hiddens_by_batch = [
            target_model_output0.mtp_collector.spec_hidden,
            target_model_output1.mtp_collector.spec_hidden,
        ]
        draft_token_ids_by_batch = [next_token_ids0, next_token_ids1]

        for draft_model in self.backend.draft_models:
            prepare_mtp_prefill_inputs(
                model_input=target_model_input0,
                b_next_token_ids=draft_token_ids_by_batch[0],
                mtp_draft_input_hiddens=draft_hiddens_by_batch[0],
            )
            prepare_mtp_prefill_inputs(
                model_input=target_model_input1,
                b_next_token_ids=draft_token_ids_by_batch[1],
                mtp_draft_input_hiddens=draft_hiddens_by_batch[1],
            )
            draft_outputs = draft_model.microbatch_overlap_prefill(
                target_model_input0,
                target_model_input1,
            )
            for batch_index, draft_output in enumerate(draft_outputs):
                draft_hiddens_by_batch[batch_index] = draft_output.mtp_collector.spec_hidden
                draft_token_ids_by_batch[batch_index] = self.backend._gen_argmax_token_ids(draft_output)

    def propose_next(
        self,
        main_model_input: ModelInput,
        main_model_output: ModelOutput,
        next_token_ids: torch.Tensor,
        b_req_mtp_start_loc: torch.Tensor,
        draft_step: int,
        accept_len: torch.Tensor | None = None,
    ) -> SpecProposal:
        verify_row_count = int(next_token_ids.shape[0])
        draft_token_ids = next_token_ids
        draft_hidden = main_model_output.mtp_collector.spec_hidden
        proposal_token_ids = next_token_ids.new_empty((verify_row_count, draft_step + 1))
        proposal_token_ids[:, 0] = next_token_ids
        schedule_scores = (
            torch.empty(
                (verify_row_count, draft_step),
                dtype=torch.float32,
                device=next_token_ids.device,
            )
            if self.enable_dynmaic_mtp
            else None
        )

        for step in range(draft_step):
            draft_model = self.backend.draft_models[step]
            main_model_input.input_ids = draft_token_ids
            main_model_input.mtp_draft_input_hiddens = draft_hidden
            draft_output = draft_model.forward(main_model_input)
            draft_hidden = draft_output.mtp_collector.spec_hidden
            if self.enable_dynmaic_mtp:
                draft_token_ids, draft_token_probs = self.backend._gen_argmax_token_ids_and_prob(draft_output)
                schedule_scores[:, step] = draft_token_probs
            else:
                draft_token_ids = self.backend._gen_argmax_token_ids(draft_output)
            proposal_token_ids[:, step + 1] = draft_token_ids

        return SpecProposal(
            token_ids=proposal_token_ids,
            extra_mem_indexes_cpu=None,
            schedule_scores=schedule_scores,
        )
