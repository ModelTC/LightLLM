from __future__ import annotations

import copy

import torch

from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
from lightllm.server.router.model_infer.mtp_speculative import utils as mtp_utils
from lightllm.server.router.model_infer.mtp_speculative.proposers.base import (
    BaseSpecProposer,
    MtpMemIndexesToFree,
)
from lightllm.server.router.model_infer.mtp_speculative.proposers.proposal_type import (
    DFlashSpecProposal,
)
from lightllm.server.router.model_infer.pin_mem_manager import g_pin_mem_manager


class DFlashProposer(BaseSpecProposer):
    """DFlash block-diffusion proposer.

    The drafter predicts a complete token block in one parallel forward from
    the accepted-tail anchor and mask-token positions.
    """

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

        target_hidden = target_model_output.mtp_collector.spec_hidden
        assert target_hidden is not None
        assert target_hidden.shape[0] == target_model_input.input_ids.shape[0]
        if target_hidden.numel() == 0:
            return

        # DFlash prefill 直接复用 target prompt 的 token 布局，并注入 target
        # hidden 初始化唯一 draft model 的 KV。使用浅副本避免在 target 输入上
        # 保留 draft 专用状态。
        draft_input = copy.copy(target_model_input)
        draft_input.mtp_draft_input_hiddens = target_hidden
        self.backend.draft_models[0].forward(draft_input)

    @torch.no_grad()
    def propose_next(
        self,
        target_model_input: ModelInput,
        target_model_output: ModelOutput,
        target_next_token_ids: torch.Tensor,
        b_req_mtp_start_loc: torch.Tensor,
        draft_step: int,
        accept_len: torch.Tensor | None = None,
    ) -> DFlashSpecProposal:
        """提交 target verify KV，并生成下一轮 DFlash block proposal。

        首次 draft forward 复用完整 target verify 布局，把本轮所有验证行的
        target hidden 写入 draft KV。随后每个请求以最后接受的 token 作为
        anchor，在其后填充 mask token，并通过一次 non-causal block forward
        并行生成完整候选块。
        """

        req_num = int(b_req_mtp_start_loc.shape[0])
        draft_model = self.backend.draft_models[0]
        block_size = int(draft_model.block_size)

        assert draft_step > 0, "DFlash requires draft_step to be greater than 0 to maintain draft KV state"
        assert draft_step <= block_size
        assert not target_model_input.is_prefill
        assert accept_len is not None
        assert accept_len.shape == (req_num,)
        assert target_next_token_ids.shape[0] == target_model_input.batch_size
        assert target_model_output.mtp_collector.spec_hidden is not None
        assert target_model_output.mtp_collector.spec_hidden.shape[0] == target_model_input.batch_size
        assert target_model_input.b_position_delta is not None
        assert len(self.backend.draft_models) == 1

        accepted_tail_rows = (b_req_mtp_start_loc + accept_len - 1).long()

        # target verify 的行布局和 mem_indexes 对应本轮所有被验证 token。
        # 附加 target hidden 后执行一次 draft forward，将这些行提交到 DFlash
        # KV cache；浅副本保证 target_model_input 本身保持不变。
        verify_draft_input = copy.copy(target_model_input)
        verify_draft_input.mtp_draft_input_hiddens = target_model_output.mtp_collector.spec_hidden
        draft_model.forward(verify_draft_input)

        # 每个请求始终展开完整 block，未被本轮 proposal 返回的 block 尾部仍会
        # 参与 parallel forward。所有临时 KV slot 在 verify 后通过 proposal
        # 统一释放。
        extra_mem_indexes_cpu = mtp_utils.alloc_mem_indexes(req_num * block_size)
        block_input_ids = target_next_token_ids.new_full(
            (req_num * block_size,),
            fill_value=draft_model.mask_token_id,
        )
        block_input_ids[::block_size] = target_next_token_ids.index_select(0, accepted_tail_rows)

        block_offsets = torch.arange(
            block_size,
            dtype=target_model_input.b_seq_len.dtype,
            device=target_next_token_ids.device,
        )
        draft_input = copy.copy(target_model_input)
        draft_input.input_ids = block_input_ids
        draft_input.mtp_draft_input_hiddens = None
        draft_input.total_token_num = req_num * block_size
        draft_input.batch_size = draft_input.total_token_num
        draft_input.max_q_seq_len = 1
        draft_input.max_kv_seq_len = target_model_input.max_kv_seq_len + block_size
        draft_input.b_req_idx = (
            target_model_input.b_req_idx.index_select(0, accepted_tail_rows).repeat_interleave(block_size).contiguous()
        )
        draft_input.b_mtp_index = g_pin_mem_manager.get_const_gpu_tensor(
            key="dflash_decode_b_mtp_index",
            shape=draft_input.b_req_idx.shape,
            fill_value=0,
            dtype=target_model_input.b_mtp_index.dtype,
        )
        draft_input.b_seq_len = (
            (target_model_input.b_seq_len.index_select(0, accepted_tail_rows)[:, None] + block_offsets[None, :] + 1)
            .reshape(-1)
            .contiguous()
        )
        draft_input.b_position_delta = (
            target_model_input.b_position_delta.index_select(0, accepted_tail_rows)
            .repeat_interleave(block_size)
            .contiguous()
        )
        draft_input.b_shared_seq_len = (
            target_model_input.b_shared_seq_len.index_select(0, accepted_tail_rows)
            .repeat_interleave(block_size)
            .contiguous()
        )
        draft_input.b_shared_radix_node_id = (
            target_model_input.b_shared_radix_node_id.index_select(0, accepted_tail_rows)
            .repeat_interleave(block_size)
            .contiguous()
        )
        draft_input.mem_indexes = extra_mem_indexes_cpu.cuda(non_blocking=True)
        draft_input.mem_indexes_cpu = None
        draft_input.multimodal_params = [{"images": [], "audios": []} for _ in range(draft_input.batch_size)]
        draft_output = draft_model.forward(draft_input)

        if self.enable_dynmaic_mtp:
            flat_draft_token_ids, flat_draft_token_probs = self.backend._gen_argmax_token_ids_and_prob(draft_output)
        else:
            flat_draft_token_ids = self.backend._gen_argmax_token_ids(draft_output)
        assert flat_draft_token_ids.numel() == req_num * block_size
        block_draft_token_ids = flat_draft_token_ids.reshape(req_num, block_size)
        proposal_token_ids = block_draft_token_ids[:, :draft_step].contiguous()

        schedule_scores = None
        if self.enable_dynmaic_mtp:
            assert flat_draft_token_probs.numel() == req_num * block_size
            block_draft_token_probs = flat_draft_token_probs.reshape(req_num, block_size)
            schedule_scores = block_draft_token_probs[:, :draft_step].float().contiguous()
        return DFlashSpecProposal(
            token_ids=proposal_token_ids,
            extra_mem_indexes_cpu=[MtpMemIndexesToFree(mem_indexes_cpu=extra_mem_indexes_cpu)],
            schedule_scores=schedule_scores,
        )
