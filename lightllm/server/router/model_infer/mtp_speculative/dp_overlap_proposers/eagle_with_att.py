import copy

import torch

from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
from lightllm.common.basemodel.triton_kernel.gen_mtp_prefill_params import gen_mtp_new_input_ids
from lightllm.server.router.model_infer.mtp_speculative import utils as mtp_utils
from lightllm.server.router.model_infer.mtp_speculative.dp_overlap_proposers.base import (
    BaseDpOverlapProposer,
)
from lightllm.server.router.model_infer.mtp_speculative.dp_overlap_proposers.utils import (
    get_dp_overlap_req_start_rows,
)
from lightllm.server.router.model_infer.mtp_speculative.proposers.base import MtpMemIndexesToFree
from lightllm.server.router.model_infer.mtp_speculative.proposers.proposal_type import EagleSpecProposal
from lightllm.server.router.model_infer.pin_mem_manager import g_pin_mem_manager


class DpOverlapEagleWithAttProposer(BaseDpOverlapProposer):
    """DP ``eagle_with_att`` proposer。"""

    def fill_draft_model_kv_state_overlap(
        self,
        target_model_input0: ModelInput,
        target_model_output0: ModelOutput,
        target_next_token_ids0: torch.Tensor,
        target_model_input1: ModelInput,
        target_model_output1: ModelOutput,
        target_next_token_ids1: torch.Tensor,
    ) -> None:
        target_model_inputs = (target_model_input0, target_model_input1)
        target_model_outputs = (target_model_output0, target_model_output1)
        target_next_token_ids = (target_next_token_ids0, target_next_token_ids1)
        assert len(self.backend.draft_models) == 1

        draft_inputs = []
        for model_input, model_output, next_token_ids in zip(
            target_model_inputs,
            target_model_outputs,
            target_next_token_ids,
        ):
            assert model_input.is_prefill
            assert model_input.b_position_delta is None
            assert next_token_ids.shape == model_input.b_req_idx.shape
            draft_input = copy.copy(model_input)
            self._prepare_eagle_prefill_inputs(
                model_input=draft_input,
                b_next_token_ids=next_token_ids,
                mtp_draft_input_hiddens=model_output.mtp_collector.spec_hidden,
            )
            draft_inputs.append(draft_input)

        self.backend.draft_models[0]._microbatch_overlap_prefill_cuda(*draft_inputs)

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
        """提交两个 target verify microbatch 的 draft KV，并生成下一轮 proposal。"""

        assert draft_step > 0, "EAGLE attention requires draft_step to be greater than 0"
        assert not target_model_input0.is_prefill
        assert not target_model_input1.is_prefill
        assert len(self.backend.draft_models) == 1

        accept_len_by_batch = (accept_len0, accept_len1)
        req_num_by_batch = (accept_len0.shape[0], accept_len1.shape[0])
        req_num = sum(req_num_by_batch)
        assert target_next_token_ids0.shape == (target_model_input0.batch_size,)
        assert target_next_token_ids1.shape == (target_model_input1.batch_size,)
        assert target_model_output0.mtp_collector.spec_hidden.shape[0] == target_model_input0.batch_size
        assert target_model_output1.mtp_collector.spec_hidden.shape[0] == target_model_input1.batch_size
        assert target_model_input0.b_position_delta is not None
        assert target_model_input1.b_position_delta is not None

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
        accepted_tail_rows_by_batch = tuple(
            (req_mtp_start_loc + batch_accept_len - 1).long()
            for req_mtp_start_loc, batch_accept_len in zip(
                b_req_mtp_start_loc,
                accept_len_by_batch,
            )
        )

        model_inputs = [copy.copy(target_model_input0), copy.copy(target_model_input1)]
        model_outputs = (target_model_output0, target_model_output1)
        target_next_token_ids_by_batch = (target_next_token_ids0, target_next_token_ids1)
        position_deltas_by_batch = tuple(model_input.b_position_delta for model_input in model_inputs)
        max_kv_seq_lens_by_batch = tuple(model_input.max_kv_seq_len for model_input in model_inputs)
        for model_input, model_output, token_ids in zip(
            model_inputs,
            model_outputs,
            target_next_token_ids_by_batch,
        ):
            model_input.input_ids = token_ids
            model_input.mtp_draft_input_hiddens = model_output.mtp_collector.spec_hidden

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

        draft_model = self.backend.draft_models[0]
        extend_outputs = draft_model._microbatch_overlap_decode_cuda(*model_inputs)
        draft_token_ids_by_batch = []
        draft_hiddens_by_batch = []
        draft_seq_lens_by_batch = []
        draft_req_indices_by_batch = []
        draft_shared_seq_lens_by_batch = []
        draft_shared_radix_node_ids_by_batch = []
        proposal_row_offsets = (0, req_num_by_batch[0])

        for batch_index, (model_input, extend_output, accepted_tail_rows, batch_req_num) in enumerate(
            zip(
                model_inputs,
                extend_outputs,
                accepted_tail_rows_by_batch,
                req_num_by_batch,
            )
        ):
            accepted_tail_output = extend_output.index_select_logits_rows(accepted_tail_rows)
            if self.enable_dynmaic_mtp:
                draft_token_ids, draft_token_probs = self._gen_argmax_token_ids_and_prob(accepted_tail_output)
                draft_token_probs = draft_token_probs.float()
            else:
                draft_token_ids = self._gen_argmax_token_ids(accepted_tail_output)

            draft_token_ids_by_batch.append(draft_token_ids)
            draft_hiddens_by_batch.append(extend_output.mtp_collector.spec_hidden.index_select(0, accepted_tail_rows))
            draft_seq_lens_by_batch.append(model_input.b_seq_len.index_select(0, accepted_tail_rows) + 1)
            draft_req_indices_by_batch.append(model_input.b_req_idx.index_select(0, accepted_tail_rows))
            draft_shared_seq_lens_by_batch.append(model_input.b_shared_seq_len.index_select(0, accepted_tail_rows))
            draft_shared_radix_node_ids_by_batch.append(
                model_input.b_shared_radix_node_id.index_select(0, accepted_tail_rows)
            )

            proposal_row_start = proposal_row_offsets[batch_index]
            proposal_row_end = proposal_row_start + batch_req_num
            proposal_token_ids[proposal_row_start:proposal_row_end, 0] = draft_token_ids
            if schedule_scores is not None:
                schedule_scores[proposal_row_start:proposal_row_end, 0] = draft_token_probs

        if draft_step == 1:
            return EagleSpecProposal(
                token_ids=proposal_token_ids,
                extra_mem_indexes_cpu=[],
                schedule_scores=schedule_scores,
            )

        for batch_index, model_input in enumerate(model_inputs):
            model_input.is_prefill = False
            model_input.batch_size = req_num_by_batch[batch_index]
            model_input.b_req_idx = draft_req_indices_by_batch[batch_index]
            model_input.b_mtp_index = torch.zeros_like(model_input.b_req_idx)
            model_input.b_seq_len = draft_seq_lens_by_batch[batch_index]
            model_input.b_position_delta = position_deltas_by_batch[batch_index].index_select(
                0, accepted_tail_rows_by_batch[batch_index]
            )
            model_input.b_shared_seq_len = draft_shared_seq_lens_by_batch[batch_index]
            model_input.b_shared_radix_node_id = draft_shared_radix_node_ids_by_batch[batch_index]
            if len(model_input.multimodal_params) != model_input.batch_size:
                empty_multimodal_params = {"images": [], "audios": []}
                model_input.multimodal_params = [empty_multimodal_params] * model_input.batch_size

        extra_mem_indexes_cpu = mtp_utils.alloc_mem_indexes(req_num * (draft_step - 1))
        extra_mem_indexes = extra_mem_indexes_cpu.to(device=target_next_token_ids0.device, non_blocking=True)

        for step in range(1, draft_step):
            mem_start = (step - 1) * req_num
            step_mem_indexes = extra_mem_indexes[mem_start : mem_start + req_num]
            mem_offset = 0
            for batch_index, model_input in enumerate(model_inputs):
                batch_req_num = req_num_by_batch[batch_index]
                model_input.input_ids = draft_token_ids_by_batch[batch_index]
                model_input.mtp_draft_input_hiddens = draft_hiddens_by_batch[batch_index]
                model_input.mem_indexes = step_mem_indexes[mem_offset : mem_offset + batch_req_num]
                model_input.max_kv_seq_len = max_kv_seq_lens_by_batch[batch_index] + step
                model_input.total_token_num = model_input.batch_size * model_input.max_kv_seq_len
                mem_offset += batch_req_num

            draft_outputs = draft_model._microbatch_overlap_decode_cuda(*model_inputs)
            for batch_index, draft_output in enumerate(draft_outputs):
                if self.enable_dynmaic_mtp:
                    draft_token_ids, draft_token_probs = self._gen_argmax_token_ids_and_prob(draft_output)
                    draft_token_probs = draft_token_probs.float()
                else:
                    draft_token_ids = self._gen_argmax_token_ids(draft_output)
                draft_token_ids_by_batch[batch_index] = draft_token_ids
                draft_hiddens_by_batch[batch_index] = draft_output.mtp_collector.spec_hidden
                draft_seq_lens_by_batch[batch_index].add_(1)

                batch_req_num = req_num_by_batch[batch_index]
                proposal_row_start = proposal_row_offsets[batch_index]
                proposal_row_end = proposal_row_start + batch_req_num
                proposal_token_ids[proposal_row_start:proposal_row_end, step] = draft_token_ids
                if schedule_scores is not None:
                    schedule_scores[proposal_row_start:proposal_row_end, step] = draft_token_probs

        return EagleSpecProposal(
            token_ids=proposal_token_ids,
            extra_mem_indexes_cpu=[MtpMemIndexesToFree(mem_indexes_cpu=extra_mem_indexes_cpu)],
            schedule_scores=schedule_scores,
        )

    def _gen_argmax_token_ids(self, model_output: ModelOutput) -> torch.Tensor:
        """生成 target vocabulary 下的候选 token；EAGLE3 会覆盖词表映射。"""

        return self.backend._gen_argmax_token_ids(model_output)

    def _gen_argmax_token_ids_and_prob(self, model_output: ModelOutput):
        """生成候选 token 及其概率；EAGLE3 会覆盖 token 的词表映射。"""

        return self.backend._gen_argmax_token_ids_and_prob(model_output)

    @staticmethod
    def _prepare_eagle_prefill_inputs(
        model_input: ModelInput,
        b_next_token_ids: torch.Tensor,
        mtp_draft_input_hiddens: torch.Tensor,
    ) -> None:
        """构造 DP-overlap EAGLE draft model 的 prefill 输入。"""

        model_input.b_is_decode_req = g_pin_mem_manager.get_const_gpu_tensor(
            key="dp_overlap_eagle_prefill_b_is_decode_req",
            shape=model_input.b_req_idx.shape,
            fill_value=False,
            dtype=torch.bool,
        )
        model_input.input_ids = gen_mtp_new_input_ids(
            input_ids=model_input.input_ids,
            b_next_token_ids=b_next_token_ids,
            b_seq_len=model_input.b_seq_len,
            b_ready_cache_len=model_input.b_ready_cache_len,
        )
        model_input.mtp_draft_input_hiddens = mtp_draft_input_hiddens
