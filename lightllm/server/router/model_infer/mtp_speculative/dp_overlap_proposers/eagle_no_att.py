import copy

import torch

from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
from lightllm.common.basemodel.triton_kernel.select_mtp_rows import select_accepted_tail_rows
from lightllm.server.router.model_infer.mtp_speculative.dp_overlap_proposers.base import (
    BaseDpOverlapProposer,
)
from lightllm.server.router.model_infer.mtp_speculative.dp_overlap_proposers.utils import (
    get_dp_overlap_req_start_rows,
)
from lightllm.server.router.model_infer.mtp_speculative.proposers.proposal_type import (
    EagleSpecProposal,
)


class DpOverlapEagleNoAttProposer(BaseDpOverlapProposer):
    """DP ``eagle_no_att`` proposer。"""

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
        target_next_token_ids0: torch.Tensor,
        accept_len0: torch.Tensor,
        target_model_input1: ModelInput,
        target_model_output1: ModelOutput,
        target_next_token_ids1: torch.Tensor,
        accept_len1: torch.Tensor,
        draft_step: int,
    ) -> EagleSpecProposal:
        accept_len_by_batch = (accept_len0, accept_len1)
        req_num_by_batch = (accept_len0.shape[0], accept_len1.shape[0])
        req_num = sum(req_num_by_batch)

        proposal_token_ids = target_next_token_ids0.new_empty((req_num, draft_step))
        schedule_scores = (
            torch.empty(
                (req_num, draft_step),
                dtype=torch.float32,
                device=target_next_token_ids0.device,
            )
            if self.enable_dynmaic_mtp
            else None
        )

        if draft_step == 0:
            return EagleSpecProposal(
                token_ids=proposal_token_ids,
                extra_mem_indexes_cpu=[],
                schedule_scores=schedule_scores,
            )

        assert target_next_token_ids0.shape == (target_model_input0.batch_size,)
        assert target_next_token_ids1.shape == (target_model_input1.batch_size,)
        b_req_mtp_start_loc = (
            get_dp_overlap_req_start_rows(
                b_mtp_index=target_model_input0.b_mtp_index,
                req_num=req_num_by_batch[0],
            ),
            get_dp_overlap_req_start_rows(
                b_mtp_index=target_model_input1.b_mtp_index,
                req_num=req_num_by_batch[1],
            ),
        )

        model_inputs = (target_model_input0, target_model_input1)
        model_outputs = (target_model_output0, target_model_output1)
        target_next_token_ids_by_batch = (target_next_token_ids0, target_next_token_ids1)
        draft_inputs = []
        draft_token_ids_by_batch = []
        draft_hiddens_by_batch = []
        for model_input, model_output, token_ids, req_mtp_start_loc, batch_accept_len, batch_req_num in zip(
            model_inputs,
            model_outputs,
            target_next_token_ids_by_batch,
            b_req_mtp_start_loc,
            accept_len_by_batch,
            req_num_by_batch,
        ):
            selected_rows = select_accepted_tail_rows(
                b_req_mtp_start_loc=req_mtp_start_loc,
                accept_len=batch_accept_len,
                input_ids=token_ids,
                hidden=model_output.mtp_collector.spec_hidden,
                b_req_idx=model_input.b_req_idx,
                b_mtp_index=model_input.b_mtp_index,
                b_seq_len=model_input.b_seq_len,
                mem_indexes=model_input.mem_indexes,
                b_shared_seq_len=model_input.b_shared_seq_len,
                b_shared_radix_node_id=model_input.b_shared_radix_node_id,
                b_position_delta=model_input.b_position_delta,
            )
            draft_input = copy.copy(model_input)
            draft_input.batch_size = batch_req_num
            draft_input.input_ids = selected_rows.input_ids
            draft_input.b_req_idx = selected_rows.b_req_idx
            draft_input.b_mtp_index = selected_rows.b_mtp_index
            draft_input.b_seq_len = selected_rows.b_seq_len
            draft_input.mem_indexes = selected_rows.mem_indexes
            draft_input.b_shared_seq_len = selected_rows.b_shared_seq_len
            draft_input.b_shared_radix_node_id = selected_rows.b_shared_radix_node_id
            draft_input.b_position_delta = selected_rows.b_position_delta
            draft_input.mem_indexes_cpu = None
            draft_input.multimodal_params = [{"images": [], "audios": []} for _ in range(batch_req_num)]
            draft_inputs.append(draft_input)
            draft_token_ids_by_batch.append(selected_rows.input_ids)
            draft_hiddens_by_batch.append(selected_rows.hidden)

        proposal_row_offsets = (0, req_num_by_batch[0])
        draft_model = self.backend.draft_models[0]
        for step in range(draft_step):
            for batch_index, draft_input in enumerate(draft_inputs):
                draft_input.input_ids = draft_token_ids_by_batch[batch_index]
                draft_input.mtp_draft_input_hiddens = draft_hiddens_by_batch[batch_index]

            draft_outputs = draft_model._microbatch_overlap_decode_cuda(*draft_inputs)
            for batch_index, draft_output in enumerate(draft_outputs):
                if self.enable_dynmaic_mtp:
                    draft_token_ids, draft_token_probs = self.backend._gen_argmax_token_ids_and_prob(draft_output)
                    draft_token_probs = draft_token_probs.float()
                else:
                    draft_token_ids = self.backend._gen_argmax_token_ids(draft_output)
                draft_token_ids_by_batch[batch_index] = draft_token_ids
                draft_hiddens_by_batch[batch_index] = draft_output.mtp_collector.spec_hidden

                batch_req_num = req_num_by_batch[batch_index]
                proposal_row_start = proposal_row_offsets[batch_index]
                proposal_row_end = proposal_row_start + batch_req_num
                proposal_token_ids[proposal_row_start:proposal_row_end, step] = draft_token_ids
                if schedule_scores is not None:
                    schedule_scores[proposal_row_start:proposal_row_end, step] = draft_token_probs

        return EagleSpecProposal(
            token_ids=proposal_token_ids,
            extra_mem_indexes_cpu=[],
            schedule_scores=schedule_scores,
        )
