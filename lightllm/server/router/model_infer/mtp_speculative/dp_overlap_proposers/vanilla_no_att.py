import copy

import torch

from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
from lightllm.server.router.model_infer.mtp_speculative.dp_overlap_proposers.base import BaseDpOverlapProposer
from lightllm.server.router.model_infer.mtp_speculative.proposers.proposal_type import VanillaSpecProposal


class DpOverlapVanillaNoAttProposer(BaseDpOverlapProposer):
    """DP ``vanilla_no_att`` proposer。"""

    def fill_draft_model_kv_state_overlap(
        self,
        target_model_input0: ModelInput,
        target_model_output0: ModelOutput,
        target_next_token_ids0: torch.Tensor,
        target_model_input1: ModelInput,
        target_model_output1: ModelOutput,
        target_next_token_ids1: torch.Tensor,
    ) -> None:
        pass

    def propose_next_overlap(
        self,
        target_model_input0: ModelInput,
        target_model_output0: ModelOutput,
        target_model_input1: ModelInput,
        target_model_output1: ModelOutput,
        target_next_token_ids: torch.Tensor,
        real_verify_rows0: int,
        real_verify_rows1: int,
        accept_len: torch.Tensor,
        draft_step: int,
    ) -> VanillaSpecProposal:
        verify_width = self.backend.max_draft_step + 1
        real_verify_rows = (int(real_verify_rows0), int(real_verify_rows1))
        req_num_by_batch = tuple(row_count // verify_width for row_count in real_verify_rows)
        assert accept_len.shape == (sum(req_num_by_batch),)

        target_next_token_ids0 = self._build_padded_next_token_ids(
            target_next_token_ids=target_next_token_ids,
            batch_size=target_model_input0.batch_size,
            real_verify_rows=real_verify_rows0,
            device=target_model_input0.b_req_idx.device,
            source_start=0,
        )
        target_next_token_ids1 = self._build_padded_next_token_ids(
            target_next_token_ids=target_next_token_ids,
            batch_size=target_model_input1.batch_size,
            real_verify_rows=real_verify_rows1,
            device=target_model_input1.b_req_idx.device,
            source_start=real_verify_rows0,
        )
        req_start_rows = (
            torch.arange(0, real_verify_rows0, verify_width, device=target_next_token_ids0.device),
            torch.arange(0, real_verify_rows1, verify_width, device=target_next_token_ids1.device),
        )
        accepted_tail_rows = (
            req_start_rows[0] + accept_len[: req_num_by_batch[0]] - 1,
            req_start_rows[1] + accept_len[req_num_by_batch[0] : sum(req_num_by_batch)] - 1,
        )
        model_inputs = [copy.copy(target_model_input0), copy.copy(target_model_input1)]
        draft_token_ids = [target_next_token_ids0, target_next_token_ids1]
        draft_hiddens = [
            target_model_output0.mtp_collector.spec_hidden,
            target_model_output1.mtp_collector.spec_hidden,
        ]
        proposal_token_ids = target_next_token_ids0.new_empty((sum(req_num_by_batch), draft_step))
        req_offset = req_num_by_batch[0]

        for step in range(draft_step):
            for batch_index, model_input in enumerate(model_inputs):
                model_input.input_ids = draft_token_ids[batch_index]
                model_input.mtp_draft_input_hiddens = draft_hiddens[batch_index]

            draft_outputs = self.backend.draft_models[step]._microbatch_overlap_decode_cuda(*model_inputs)
            for batch_index, draft_output in enumerate(draft_outputs):
                draft_hiddens[batch_index] = draft_output.mtp_collector.spec_hidden
                draft_token_ids[batch_index] = self.backend._gen_argmax_token_ids(draft_output)

            proposal_token_ids[:req_offset, step] = draft_token_ids[0].index_select(0, accepted_tail_rows[0].long())
            proposal_token_ids[req_offset:, step] = draft_token_ids[1].index_select(0, accepted_tail_rows[1].long())

        return VanillaSpecProposal(
            token_ids=proposal_token_ids,
            extra_mem_indexes_cpu=[],
            schedule_scores=None,
        )
