import copy

import torch

from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
from lightllm.common.basemodel.triton_kernel.gen_mtp_prefill_params import gen_mtp_new_input_ids
from lightllm.server.router.model_infer.mtp_speculative.dp_proposers.base import BaseDpProposer
from lightllm.server.router.model_infer.mtp_speculative.proposers.vanilla_utils import (
    VanillaSpecProposal,
    propose_next_chained_mtp,
)
from lightllm.server.router.model_infer.pin_mem_manager import g_pin_mem_manager


class DpVanillaWithAttProposer(BaseDpProposer):
    """普通 DP ``vanilla_with_att`` proposer。"""

    def fill_draft_model_kv_state(
        self,
        target_model_input: ModelInput,
        target_model_output: ModelOutput,
        target_next_token_ids: torch.Tensor,
    ) -> None:
        assert target_model_input.is_prefill
        assert target_model_input.b_position_delta is None
        assert target_next_token_ids.shape == target_model_input.b_req_idx.shape

        draft_input = copy.copy(target_model_input)
        draft_hidden = target_model_output.mtp_collector.spec_hidden
        draft_token_ids = target_next_token_ids
        for draft_model in self.backend.draft_models:
            draft_input = self._prepare_mtp_prefill_inputs(
                model_input=draft_input,
                b_next_token_ids=draft_token_ids,
                mtp_draft_input_hiddens=draft_hidden,
            )
            draft_output = draft_model.forward(draft_input)
            draft_hidden = draft_output.mtp_collector.spec_hidden
            draft_token_ids = self.backend._gen_argmax_token_ids(draft_output)

    def propose_next(
        self,
        target_model_input: ModelInput,
        target_model_output: ModelOutput,
        target_next_token_ids: torch.Tensor,
        b_req_mtp_start_loc: torch.Tensor,
        draft_step: int,
        accept_len: torch.Tensor | None = None,
    ) -> VanillaSpecProposal:
        return propose_next_chained_mtp(
            self,
            target_model_input,
            target_model_output,
            target_next_token_ids,
            b_req_mtp_start_loc,
            draft_step,
            accept_len,
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
            key="dp_vanilla_mtp_prefill_b_is_decode_req",
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
