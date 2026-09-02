import copy

import torch

from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
from lightllm.common.basemodel.triton_kernel.gen_mtp_prefill_params import gen_mtp_new_input_ids
from lightllm.common.basemodel.triton_kernel.select_mtp_rows import select_accepted_tail_rows
from lightllm.server.router.model_infer.mtp_speculative import utils as mtp_utils
from lightllm.server.router.model_infer.mtp_speculative.proposers.base import (
    BaseSpecProposer,
    MtpMemIndexesToFree,
)
from lightllm.server.router.model_infer.mtp_speculative.proposers.proposal_type import EagleSpecProposal
from lightllm.server.router.model_infer.pin_mem_manager import g_pin_mem_manager


class EagleWithAttProposer(BaseSpecProposer):
    """使用 attention KV cache 的 EAGLE proposer。"""

    def fill_draft_model_kv_state(
        self,
        target_model_input: ModelInput,
        target_model_output: ModelOutput,
        target_next_token_ids: torch.Tensor,
    ) -> None:
        assert target_model_input.is_prefill
        assert target_model_input.b_position_delta is None
        assert target_next_token_ids.shape == target_model_input.b_req_idx.shape
        assert len(self.backend.draft_models) == 1

        # EAGLE 只有一个递归复用的 draft model。用 target prefill 的 token、
        # next token 和 hidden 构造左移一位的 draft 输入，将 prompt KV 写入
        # draft cache。使用浅副本避免覆盖后续流程仍可能读取的 target 输入。
        draft_input = copy.copy(target_model_input)
        self._prepare_eagle_prefill_inputs(
            model_input=draft_input,
            b_next_token_ids=target_next_token_ids,
            mtp_draft_input_hiddens=target_model_output.mtp_collector.spec_hidden,
        )
        self.backend.draft_models[0].forward(draft_input)

    def propose_next(
        self,
        target_model_input: ModelInput,
        target_model_output: ModelOutput,
        target_next_token_ids: torch.Tensor,
        b_req_mtp_start_loc: torch.Tensor,
        draft_step: int,
        accept_len: torch.Tensor | None = None,
    ) -> EagleSpecProposal:
        """提交验证结果对应的 draft KV，并递归生成下一轮 EAGLE proposal。

        第一次 draft forward 复用完整 target verify 布局，把本轮所有验证行
        写入 draft KV；同时从每个请求最后接受的位置生成第一个候选 token。
        后续级别只保留每个请求一行，使用临时 KV slot 递归生成剩余候选。
        """

        req_num = int(b_req_mtp_start_loc.shape[0])
        proposal_token_ids_by_step = []
        schedule_scores_by_step = []

        assert draft_step > 0, "EAGLE attention requires draft_step to be greater than 0 to maintain draft KV state"
        assert not target_model_input.is_prefill
        assert accept_len is not None
        assert accept_len.shape == (req_num,)
        assert target_next_token_ids.shape[0] == target_model_input.batch_size
        assert target_model_output.mtp_collector.spec_hidden.shape[0] == target_model_input.batch_size
        assert target_model_input.b_position_delta is not None
        assert len(self.backend.draft_models) == 1

        accepted_tail_rows = (b_req_mtp_start_loc + accept_len - 1).long()
        draft_model = self.backend.draft_models[0]

        # target verify 的 mem_indexes 正是已验证 token 应写入的 KV slot。
        # 仅替换 token 和 hidden，其他布局 tensor 与 target 共享；浅副本保证
        # draft forward 不会改变 target_model_input 对象自身的字段。
        verify_draft_input = copy.copy(target_model_input)
        verify_draft_input.input_ids = target_next_token_ids
        verify_draft_input.mtp_draft_input_hiddens = target_model_output.mtp_collector.spec_hidden
        extend_output = draft_model.forward(verify_draft_input)

        # 只在 req_num 行 logits 上进行 argmax，避免为未接受的 verify 行执行
        # vocabulary reduction。第一列 proposal 来自每个请求的 accepted tail。
        accepted_tail_output = extend_output.index_select_logits_rows(accepted_tail_rows)
        if self.enable_dynmaic_mtp:
            draft_token_ids, draft_token_probs = self._gen_argmax_token_ids_and_prob(accepted_tail_output)
            schedule_scores_by_step.append(draft_token_probs.float().unsqueeze(1))
        else:
            draft_token_ids = self._gen_argmax_token_ids(accepted_tail_output)
        proposal_token_ids_by_step.append(draft_token_ids.unsqueeze(1))

        if draft_step == 1:
            return EagleSpecProposal(
                token_ids=torch.cat(proposal_token_ids_by_step, dim=1),
                extra_mem_indexes_cpu=[],
                schedule_scores=torch.cat(schedule_scores_by_step, dim=1) if self.enable_dynmaic_mtp else None,
            )

        # 后续递归每步、每请求各写一个临时 KV。proposal 在 verify 完成后
        # 统一释放这些 slot，因此同时保留 CPU 索引用于资源回收。
        extra_mem_indexes_cpu = mtp_utils.alloc_mem_indexes(req_num * (draft_step - 1))
        extra_mem_indexes = extra_mem_indexes_cpu.cuda(non_blocking=True)

        # 一次 Triton kernel 合并抽取 accepted-tail 的 hidden、请求索引、
        # 序列长度、position delta 和共享 radix 元数据，构造 req_num 行的
        # 单 token decode 输入。通用算子同时返回 accepted-tail input ids，
        # 因而这里传入与 verify 布局一致的 target_next_token_ids；EAGLE
        # 递归 token 来自上面的 draft logits，不使用 selected_rows.input_ids。
        selected_rows = select_accepted_tail_rows(
            b_req_mtp_start_loc=b_req_mtp_start_loc,
            accept_len=accept_len,
            input_ids=target_next_token_ids,
            hidden=extend_output.mtp_collector.spec_hidden,
            b_req_idx=target_model_input.b_req_idx,
            b_mtp_index=target_model_input.b_mtp_index,
            b_seq_len=target_model_input.b_seq_len,
            mem_indexes=target_model_input.mem_indexes,
            b_shared_seq_len=target_model_input.b_shared_seq_len,
            b_shared_radix_node_id=target_model_input.b_shared_radix_node_id,
            b_position_delta=target_model_input.b_position_delta,
        )
        draft_hidden = selected_rows.hidden
        draft_seq_lens = selected_rows.b_seq_len + 1
        max_kv_seq_len = target_model_input.max_kv_seq_len
        draft_input = copy.copy(target_model_input)
        draft_input.is_prefill = False
        draft_input.batch_size = req_num
        draft_input.b_req_idx = selected_rows.b_req_idx
        draft_input.b_mtp_index = g_pin_mem_manager.get_const_gpu_tensor(
            key="eagle_with_att_decode_b_mtp_index",
            shape=selected_rows.b_req_idx.shape,
            fill_value=0,
            dtype=target_model_input.b_mtp_index.dtype,
        )
        draft_input.b_seq_len = draft_seq_lens
        draft_input.b_position_delta = selected_rows.b_position_delta
        draft_input.b_shared_seq_len = selected_rows.b_shared_seq_len
        draft_input.b_shared_radix_node_id = selected_rows.b_shared_radix_node_id
        draft_input.mem_indexes_cpu = None
        draft_input.multimodal_params = [{"images": [], "audios": []} for _ in range(req_num)]

        for step in range(1, draft_step):
            mem_start = (step - 1) * req_num
            draft_input.input_ids = draft_token_ids
            draft_input.mtp_draft_input_hiddens = draft_hidden
            draft_input.mem_indexes = extra_mem_indexes[mem_start : mem_start + req_num]
            draft_input.max_kv_seq_len = max_kv_seq_len + step
            draft_input.total_token_num = req_num * draft_input.max_kv_seq_len
            draft_output = draft_model.forward(draft_input)
            draft_hidden = draft_output.mtp_collector.spec_hidden

            if self.enable_dynmaic_mtp:
                draft_token_ids, draft_token_probs = self._gen_argmax_token_ids_and_prob(draft_output)
                schedule_scores_by_step.append(draft_token_probs.float().unsqueeze(1))
            else:
                draft_token_ids = self._gen_argmax_token_ids(draft_output)
            proposal_token_ids_by_step.append(draft_token_ids.unsqueeze(1))
            draft_seq_lens.add_(1)

        proposal_token_ids = torch.cat(proposal_token_ids_by_step, dim=1)
        schedule_scores = torch.cat(schedule_scores_by_step, dim=1) if self.enable_dynmaic_mtp else None
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

    def _prepare_eagle_prefill_inputs(
        self,
        model_input: ModelInput,
        b_next_token_ids: torch.Tensor,
        mtp_draft_input_hiddens: torch.Tensor,
    ) -> ModelInput:
        """构造 EAGLE With-Att draft model 的 prompt prefill 输入。

        EAGLE 的 pre-layer 同时使用 token embedding 和 target hidden。为了让
        draft token 与 target hidden 在预测位置上对齐，本方法会在每个请求
        当前 query 区间内将 ``input_ids`` 左移一位：移除区间首 token，并
        将该请求的 ``b_next_token_ids`` 追加到区间末尾。Chunked prefill
        时，已经位于 KV cache 中的 prefix 不参与移动。

        此外会把 ``b_is_decode_req`` 设置为缓存的全 False GPU 张量，使
        mixed-prefill 输入整理逻辑保留这里生成的 token；target model 输出的
        hidden 则通过 ``mtp_draft_input_hiddens`` 传给 EAGLE pre-layer。

        参数：
            model_input: draft prompt prefill 输入。本方法原地更新其
                ``input_ids``、``b_is_decode_req`` 和 hidden 字段。
            b_next_token_ids: 每个请求追加到当前 query 末尾的 next token，
                shape 为 ``[batch_size]``。
            mtp_draft_input_hiddens: target prefill 为当前 query 产生的 hidden，
                第一维与扁平 ``input_ids`` 对齐。

        返回：
            更新后的 ``model_input``，与传入对象相同。

        示例：
            若两个请求当前 query token 分别为 ``[10, 11, 12]`` 和
            ``[20, 21, 22]``，追加 token 为 ``[13, 23]``，则输入从
            ``[10, 11, 12, 20, 21, 22]`` 变为
            ``[11, 12, 13, 21, 22, 23]``。若请求存在 cached prefix，移动
            仍只发生在当前 query，cache 中的历史 token 和 KV 均保持不变。
        """
        model_input.b_is_decode_req = g_pin_mem_manager.get_const_gpu_tensor(
            key="eagle_with_att_prefill_b_is_decode_req",
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
