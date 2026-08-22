import copy

import torch

from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
from lightllm.common.basemodel.triton_kernel.select_mtp_rows import (
    select_accepted_tail_rows,
)
from lightllm.server.router.model_infer.mtp_speculative.proposers.base import BaseSpecProposer
from lightllm.server.router.model_infer.mtp_speculative.proposers.proposal_type import EagleSpecProposal


class EagleNoAttProposer(BaseSpecProposer):
    """不使用 attention KV cache 的 EAGLE proposer。"""

    def fill_draft_model_kv_state(
        self,
        target_model_input: ModelInput,
        target_model_output: ModelOutput,
        target_next_token_ids: torch.Tensor,
    ) -> None:
        pass

    def propose_next(
        self,
        target_model_input: ModelInput,
        target_model_output: ModelOutput,
        target_next_token_ids: torch.Tensor,
        b_req_mtp_start_loc: torch.Tensor,
        draft_step: int,
        accept_len: torch.Tensor | None = None,
    ) -> EagleSpecProposal:
        req_num = int(b_req_mtp_start_loc.shape[0])
        proposal_token_ids_by_step = []
        schedule_scores_by_step = []

        if draft_step == 0:
            return EagleSpecProposal(
                token_ids=target_next_token_ids.new_empty((req_num, 0)),
                extra_mem_indexes_cpu=[],
                schedule_scores=(
                    torch.empty(
                        (req_num, 0),
                        dtype=torch.float32,
                        device=target_next_token_ids.device,
                    )
                    if self.enable_dynmaic_mtp
                    else None
                ),
            )

        assert accept_len is not None
        # EAGLE No-Att 不维护 draft KV cache。target verification 的输出布局为
        # [verify_batch_size, ...]，而下一轮 proposal 每个请求只需要最后一个
        # 接受位置的 token 和 hidden。这里先将所有相关输入压缩到 [req_num, ...]，
        # 后续各级递归均只计算这 req_num 行。
        selected_rows = select_accepted_tail_rows(
            b_req_mtp_start_loc=b_req_mtp_start_loc,
            accept_len=accept_len,
            input_ids=target_next_token_ids,
            hidden=target_model_output.mtp_collector.spec_hidden,
            b_req_idx=target_model_input.b_req_idx,
            b_mtp_index=target_model_input.b_mtp_index,
            b_seq_len=target_model_input.b_seq_len,
            mem_indexes=target_model_input.mem_indexes,
            b_shared_seq_len=target_model_input.b_shared_seq_len,
            b_shared_radix_node_id=target_model_input.b_shared_radix_node_id,
            b_position_delta=target_model_input.b_position_delta,
        )
        draft_token_ids = selected_rows.input_ids
        draft_hidden = selected_rows.hidden
        draft_input = copy.copy(target_model_input)
        draft_input.batch_size = req_num
        draft_input.input_ids = draft_token_ids
        draft_input.b_req_idx = selected_rows.b_req_idx
        draft_input.b_mtp_index = selected_rows.b_mtp_index
        draft_input.b_seq_len = selected_rows.b_seq_len
        draft_input.mem_indexes = selected_rows.mem_indexes
        draft_input.b_shared_seq_len = selected_rows.b_shared_seq_len
        draft_input.b_shared_radix_node_id = selected_rows.b_shared_radix_node_id
        draft_input.b_position_delta = selected_rows.b_position_delta
        draft_input.mem_indexes_cpu = None
        draft_input.multimodal_params = [{"images": [], "audios": []} for _ in range(req_num)]

        # EAGLE 使用同一个 draft model 递归生成多个 token。每一级将上一级
        # 生成的 token 与 hidden 作为下一次输入；No-Att 模式没有 KV 状态，
        # 因此无需分配临时 mem indexes 或修改序列长度。
        draft_model = self.backend.draft_models[0]
        for _ in range(draft_step):
            draft_input.input_ids = draft_token_ids
            draft_input.mtp_draft_input_hiddens = draft_hidden
            draft_output = draft_model.forward(draft_input)
            draft_hidden = draft_output.mtp_collector.spec_hidden

            if self.enable_dynmaic_mtp:
                draft_token_ids, draft_token_probs = self.backend._gen_argmax_token_ids_and_prob(draft_output)
                schedule_scores_by_step.append(draft_token_probs.float().unsqueeze(1))
            else:
                draft_token_ids = self.backend._gen_argmax_token_ids(draft_output)
            proposal_token_ids_by_step.append(draft_token_ids.unsqueeze(1))

        proposal_token_ids = torch.cat(proposal_token_ids_by_step, dim=1)
        schedule_scores = torch.cat(schedule_scores_by_step, dim=1) if self.enable_dynmaic_mtp else None
        return EagleSpecProposal(
            token_ids=proposal_token_ids,
            extra_mem_indexes_cpu=[],
            schedule_scores=schedule_scores,
        )
