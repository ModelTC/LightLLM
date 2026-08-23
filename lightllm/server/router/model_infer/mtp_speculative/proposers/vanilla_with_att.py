import copy

import torch

from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
from lightllm.common.basemodel.triton_kernel.gen_mtp_prefill_params import gen_mtp_new_input_ids
from lightllm.common.basemodel.triton_kernel.build_chained_mtp_decode_input import (
    build_chained_mtp_decode_input_inplace,
)
from lightllm.server.router.model_infer.mtp_speculative.proposers.base import BaseSpecProposer
from lightllm.server.router.model_infer.mtp_speculative.proposers.proposal_type import VanillaSpecProposal
from lightllm.server.router.model_infer.pin_mem_manager import g_pin_mem_manager


class VanillaWithAttProposer(BaseSpecProposer):
    """使用 attention KV cache 的 Vanilla chained MTP proposer。"""

    def fill_draft_model_kv_state(
        self,
        target_model_input: ModelInput,
        target_model_output: ModelOutput,
        target_next_token_ids: torch.Tensor,
    ) -> None:
        assert target_model_input.is_prefill
        assert target_model_input.b_position_delta is None
        assert target_next_token_ids.shape == target_model_input.b_req_idx.shape

        # 在局部副本上逐级左移 token 并传递 hidden，避免修改 target
        # prefill 输入。Chunked prefill 边界继续用当前级预测 token 补齐。
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
        """运行完整 Vanilla-With-Att 级联并生成下一轮 proposal。

        每一级 draft model 都维护自己所在级联位置的 attention KV，因此
        decode 时必须依次运行全部 draft model，不能动态缩短级联深度。
        每一级在完整 target verify 布局上 forward，以补齐该级对应位置的
        KV；最终只抽取每个请求最后接受位置的输出作为下一轮候选 token。
        """

        req_num = int(b_req_mtp_start_loc.shape[0])
        proposal_token_ids_by_step = []
        schedule_scores_by_step = []

        assert not target_model_input.is_prefill
        assert accept_len is not None
        assert accept_len.shape == (req_num,)
        assert target_next_token_ids.shape[0] == target_model_input.batch_size
        assert target_model_output.mtp_collector.spec_hidden.shape[0] == target_model_input.batch_size
        assert draft_step == self.backend.max_draft_step, (
            "vanilla_with_att requires the full chained draft depth: "
            f"draft_step={draft_step}, max_draft_step={self.backend.max_draft_step}"
        )
        assert len(self.backend.draft_models) == draft_step

        # target verify 布局中，同一请求的行从 b_req_mtp_start_loc 开始，
        # accept_len 包含必然接受的 target token。因此减 1 后得到本轮最后
        # 接受 token 所在的物理行，后续每一级都从这些固定行收集 proposal。
        accepted_tail_rows = (b_req_mtp_start_loc + accept_len - 1).long()

        # draft forward 需要复用 target decode 的请求、位置和 KV slot 布局，
        # 但 input_ids 与 hidden 会逐级更新。使用浅副本避免覆盖 target 输入
        # 中这两个字段；布局 tensor 仍然共享，不引入额外拷贝。
        draft_token_ids = target_next_token_ids
        draft_hidden = target_model_output.mtp_collector.spec_hidden
        draft_input = copy.copy(target_model_input)

        for step in range(draft_step):
            draft_input.input_ids = draft_token_ids
            draft_input.mtp_draft_input_hiddens = draft_hidden
            draft_output = self.backend.draft_models[step].forward(draft_input)
            draft_hidden = draft_output.mtp_collector.spec_hidden

            if self.enable_dynmaic_mtp:
                # Vanilla With-Att 没有独立的 confidence head；动态调度使用
                # 当前 draft model 选中 token 的采样概率。
                draft_token_ids, draft_token_probs = self.backend._gen_argmax_token_ids_and_prob(draft_output)
                selected_token_probs = draft_token_probs.index_select(0, accepted_tail_rows)
                schedule_scores_by_step.append(selected_token_probs.float().unsqueeze(1))
            else:
                draft_token_ids = self.backend._gen_argmax_token_ids(draft_output)
            selected_token_ids = draft_token_ids.index_select(0, accepted_tail_rows)
            proposal_token_ids_by_step.append(selected_token_ids.unsqueeze(1))

            if step + 1 < draft_step:
                # 下一层不能直接使用所有行的 draft 预测。已接受前缀继续使用
                # main/上一层输入中的真实 token，仅在 tail 行接上本级新生成
                # 的 draft token，从而形成逐级左移并覆盖尾部的级联输入。
                draft_token_ids = build_chained_mtp_decode_input_inplace(
                    input_ids=draft_input.input_ids,
                    draft_token_ids=draft_token_ids,
                    b_req_mtp_start_loc=b_req_mtp_start_loc,
                    accept_len=accept_len,
                )

        proposal_token_ids = torch.cat(proposal_token_ids_by_step, dim=1)
        schedule_scores = torch.cat(schedule_scores_by_step, dim=1) if self.enable_dynmaic_mtp else None
        return VanillaSpecProposal(
            token_ids=proposal_token_ids,
            extra_mem_indexes_cpu=[],
            schedule_scores=schedule_scores,
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
            key="vanilla_mtp_prefill_b_is_decode_req",
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
