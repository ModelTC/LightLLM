import copy

import torch

from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
from lightllm.common.basemodel.triton_kernel.gen_mtp_prefill_params import (
    gen_mtp_new_input_ids,
)
from lightllm.common.basemodel.triton_kernel.build_chained_mtp_decode_input import (
    build_chained_mtp_decode_input_inplace,
)
from lightllm.server.router.model_infer.mtp_speculative.dp_overlap_proposers.base import (
    BaseDpOverlapProposer,
)
from lightllm.server.router.model_infer.mtp_speculative.dp_overlap_proposers.utils import (
    get_dp_overlap_req_start_rows,
)
from lightllm.server.router.model_infer.mtp_speculative.proposers.proposal_type import (
    VanillaSpecProposal,
)
from lightllm.server.router.model_infer.pin_mem_manager import g_pin_mem_manager


class DpOverlapVanillaWithAttProposer(BaseDpOverlapProposer):
    """DP ``vanilla_with_att`` proposer。"""

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
        target_next_token_ids = (target_next_token_ids0, target_next_token_ids1)
        for model_input, next_token_ids in zip(target_model_inputs, target_next_token_ids):
            assert model_input.is_prefill
            assert model_input.b_position_delta is None
            assert next_token_ids.shape == model_input.b_req_idx.shape

        model_inputs = [copy.copy(model_input) for model_input in target_model_inputs]
        draft_hiddens = [
            target_model_output0.mtp_collector.spec_hidden,
            target_model_output1.mtp_collector.spec_hidden,
        ]
        draft_token_ids = list(target_next_token_ids)

        for draft_model in self.backend.draft_models:
            for batch_index, model_input in enumerate(model_inputs):
                model_inputs[batch_index] = self._prepare_mtp_prefill_inputs(
                    model_input=model_input,
                    b_next_token_ids=draft_token_ids[batch_index],
                    mtp_draft_input_hiddens=draft_hiddens[batch_index],
                )
            draft_outputs = draft_model._microbatch_overlap_prefill_cuda(*model_inputs)
            for batch_index, draft_output in enumerate(draft_outputs):
                draft_hiddens[batch_index] = draft_output.mtp_collector.spec_hidden
                draft_token_ids[batch_index] = self.backend._gen_argmax_token_ids(draft_output)

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
    ) -> VanillaSpecProposal:
        assert draft_step == self.backend.max_draft_step
        assert len(self.backend.draft_models) == draft_step

        accept_len_by_batch = (accept_len0, accept_len1)
        req_num_by_batch = (accept_len0.shape[0], accept_len1.shape[0])

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
        accepted_tail_rows = tuple(
            req_mtp_start_loc + batch_accept_len - 1
            for req_mtp_start_loc, batch_accept_len in zip(
                b_req_mtp_start_loc,
                accept_len_by_batch,
            )
        )
        model_inputs = [copy.copy(target_model_input0), copy.copy(target_model_input1)]
        draft_token_ids = [target_next_token_ids0, target_next_token_ids1]
        draft_hiddens = [
            target_model_output0.mtp_collector.spec_hidden,
            target_model_output1.mtp_collector.spec_hidden,
        ]
        proposal_token_ids = target_next_token_ids0.new_empty((sum(req_num_by_batch), draft_step))
        proposal_schedule_scores = (
            torch.empty(
                (sum(req_num_by_batch), draft_step),
                dtype=torch.float32,
                device=target_next_token_ids0.device,
            )
            if self.enable_dynmaic_mtp
            else None
        )
        req_offset = req_num_by_batch[0]

        for step in range(draft_step):
            for batch_index, model_input in enumerate(model_inputs):
                model_input.input_ids = draft_token_ids[batch_index]
                model_input.mtp_draft_input_hiddens = draft_hiddens[batch_index]

            draft_outputs = self.backend.draft_models[step]._microbatch_overlap_decode_cuda(*model_inputs)
            for batch_index, draft_output in enumerate(draft_outputs):
                draft_hiddens[batch_index] = draft_output.mtp_collector.spec_hidden
                if self.enable_dynmaic_mtp:
                    draft_token_ids[batch_index], draft_token_probs = self.backend._gen_argmax_token_ids_and_prob(
                        draft_output
                    )
                    selected_token_probs = draft_token_probs.index_select(
                        0,
                        accepted_tail_rows[batch_index].long(),
                    )
                    proposal_row_start = 0 if batch_index == 0 else req_offset
                    proposal_schedule_scores[
                        proposal_row_start : proposal_row_start + req_num_by_batch[batch_index],
                        step,
                    ] = selected_token_probs
                else:
                    draft_token_ids[batch_index] = self.backend._gen_argmax_token_ids(draft_output)

            proposal_token_ids[:req_offset, step] = draft_token_ids[0].index_select(0, accepted_tail_rows[0].long())
            proposal_token_ids[req_offset:, step] = draft_token_ids[1].index_select(0, accepted_tail_rows[1].long())

            if step + 1 < draft_step:
                for batch_index, model_input in enumerate(model_inputs):
                    draft_token_ids[batch_index] = build_chained_mtp_decode_input_inplace(
                        input_ids=model_input.input_ids,
                        draft_token_ids=draft_token_ids[batch_index],
                        b_req_mtp_start_loc=b_req_mtp_start_loc[batch_index],
                        accept_len=accept_len_by_batch[batch_index],
                    )

        return VanillaSpecProposal(
            token_ids=proposal_token_ids,
            extra_mem_indexes_cpu=[],
            schedule_scores=proposal_schedule_scores,
        )

    def _prepare_mtp_prefill_inputs(
        self,
        model_input: ModelInput,
        b_next_token_ids: torch.Tensor,
        mtp_draft_input_hiddens: torch.Tensor,
    ) -> ModelInput:
        """构造 Vanilla chained MTP 下一层 draft model 的 prefill 输入。

        每个请求当前参与计算的 query 长度为
        ``b_seq_len - b_ready_cache_len``。本方法在各请求自己的 query
        区间内将 token 左移一位，即丢弃区间首 token，并在区间尾部追加该
        请求的 ``b_next_token_ids``。这样，下一层 MTP model 看到的 token
        与上一层 model 生成的 next token 连续对齐。Chunked prefill 时只
        移动当前 chunk，已经写入 KV cache 的 prefix 不参与移动。

        同时，本方法会完成两项辅助输入设置：

        1. 将 ``b_is_decode_req`` 设置为全 False 的缓存 GPU 常量张量，确保
           mixed-prefill 逻辑保留这里显式生成的 token，而不会按 decode
           请求重新收集 token。
        2. 将上一层 model 输出的 hidden states 绑定到
           ``mtp_draft_input_hiddens``，供下一层 MTP model 使用。

        参数：
            model_input: 当前层的 prefill 输入。方法会原地更新该对象的
                ``input_ids``、``b_is_decode_req`` 和
                ``mtp_draft_input_hiddens`` 字段。
            b_next_token_ids: 每个请求需要追加到当前 query 尾部的 token，
                shape 为 ``[batch_size]``。
            mtp_draft_input_hiddens: 上一层 model 产生、供下一层 draft model
                使用的 hidden states。

        返回：
            更新后的 ``model_input``，与传入对象是同一个对象。

        示例：
            假设 ``b_seq_len=[4, 5]``、``b_ready_cache_len=[1, 2]``，则两个
            请求当前 query 长度均为 3。若扁平输入和追加 token 为：

            ``input_ids = [10, 11, 12, 20, 21, 22]``
            ``b_next_token_ids = [13, 23]``

            更新后的扁平输入为：

            ``input_ids = [11, 12, 13, 21, 22, 23]``

            两个请求已缓存的 prefix 长度分别为 1 和 2，它们对应的 token
            不在 ``input_ids`` 中，因此不会被本方法移动或重写。
        """
        model_input.b_is_decode_req = g_pin_mem_manager.get_const_gpu_tensor(
            key="dp_overlap_vanilla_mtp_prefill_b_is_decode_req",
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
        return model_input
