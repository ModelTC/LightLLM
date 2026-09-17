from __future__ import annotations

import copy

import torch

from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
from lightllm.server.router.model_infer.mtp_speculative import utils as mtp_utils
from lightllm.server.router.model_infer.pin_mem_manager import g_pin_mem_manager
from lightllm.server.router.model_infer.mtp_speculative.proposers.base import (
    BaseSpecProposer,
    MtpMemIndexesToFree,
)
from lightllm.server.router.model_infer.mtp_speculative.proposers.proposal_type import (
    DSparkSpecProposal,
)


class DSparkProposer(BaseSpecProposer):
    """DSpark semi-autoregressive parallel-block proposer.

    A parallel DFlash-style backbone generates the block features in one pass;
    a lightweight sequential Markov head adds intra-block token dependency.
    The confidence head supplies per-position scheduling scores.
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

        # DSpark prefill 直接使用 target prompt 的 token 布局和 hidden，将 prompt
        # KV 写入唯一的 parallel-block draft model。使用浅副本，避免把 draft
        # 专用 hidden 挂到后续流程仍可能读取的 target ModelInput 上。
        if getattr(self.backend.draft_models[0], "uses_windowed_draft_kv", False) is True:
            self.backend.draft_models[0].commit_features(target_model_input, target_hidden)
            return
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
    ) -> DSparkSpecProposal:
        """提交 target verify KV，并生成下一轮 DSpark block proposal。

        首次 draft forward 复用完整 target verify 布局，把本轮所有验证行的
        target hidden 写入 draft KV。随后每个请求取最后接受的 token 作为
        block anchor，并在其后填充 mask token；第二次 forward 一次生成完整
        block。Markov head 可以直接返回具有块内依赖的 token，confidence head
        则为动态 verify 提供每个 draft 位置的条件置信度。
        """

        req_num = int(b_req_mtp_start_loc.shape[0])
        draft_model = self.backend.draft_models[0]
        block_size = int(draft_model.block_size)
        schedule_scores = None

        assert draft_step > 0, "DSpark requires draft_step to be greater than 0 to maintain draft KV state"
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

        # full 模式通过 draft forward 写入 verify 行对应的 KV；windowed 模式
        # 只将接受的 target hidden 投影并写入独立窗口，不保留被拒绝的行。
        windowed = getattr(draft_model, "uses_windowed_draft_kv", False) is True
        if windowed:
            draft_model.commit_features(
                target_model_input, target_model_output.mtp_collector.spec_hidden, b_req_mtp_start_loc, accept_len
            )
        else:
            verify_draft_input = copy.copy(target_model_input)
            verify_draft_input.mtp_draft_input_hiddens = target_model_output.mtp_collector.spec_hidden
            draft_model.forward(verify_draft_input)

        # 每个请求固定展开完整 block：首行是 accepted-tail anchor，其余为 mask。
        # full 模式的临时 KV 在 verify 后释放；windowed 模式不占用 target KV slot。
        if windowed:
            extra_mem_indexes_cpu = None
            scratch_indexes = torch.full(
                (req_num * block_size,),
                draft_model.mem_manager.HOLD_TOKEN_MEMINDEX,
                dtype=torch.int32,
                device=target_next_token_ids.device,
            )
        else:
            extra_mem_indexes_cpu = mtp_utils.alloc_mem_indexes(req_num * block_size)
            scratch_indexes = extra_mem_indexes_cpu.cuda(non_blocking=True)
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
            key="dspark_decode_b_mtp_index",
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
        draft_input.mem_indexes = scratch_indexes
        draft_input.mem_indexes_cpu = None
        draft_input.multimodal_params = [{"images": [], "audios": []} for _ in range(draft_input.batch_size)]
        draft_output = draft_model.forward(draft_input)

        if draft_output.mtp_collector.draft_token_ids is None:
            flat_draft_token_ids = self.backend._gen_argmax_token_ids(draft_output)
        else:
            flat_draft_token_ids = draft_output.mtp_collector.draft_token_ids
        assert flat_draft_token_ids.numel() == req_num * block_size
        block_draft_token_ids = flat_draft_token_ids.reshape(req_num, block_size)
        proposal_token_ids = block_draft_token_ids[:, :draft_step].contiguous()

        if self.enable_dynmaic_mtp:
            confidence_logits = draft_output.mtp_collector.confidence_logits
            if confidence_logits is None:
                raise RuntimeError("DSpark dynamic verify requires confidence head logits")
            assert confidence_logits.ndim == 2
            assert confidence_logits.shape[0] == req_num
            assert confidence_logits.shape[1] >= draft_step
            # Match the clamp used by the GPU dynamic row selector before it
            # converts conditional confidence to prefix survival probability.
            schedule_scores = (
                confidence_logits[:, :draft_step]
                .sigmoid()
                .clamp(
                    min=0.01,
                    max=0.99,
                )
                .contiguous()
            )

        schedule_scores_cpu = None
        if schedule_scores is not None:
            schedule_scores_cpu = g_pin_mem_manager.async_copy_from_gpu_tensor(
                key="dspark_confidence_probs",
                gpu_tensor=schedule_scores,
            )

        return DSparkSpecProposal(
            token_ids=proposal_token_ids,
            extra_mem_indexes_cpu=(
                [MtpMemIndexesToFree(mem_indexes_cpu=extra_mem_indexes_cpu)]
                if extra_mem_indexes_cpu is not None
                else []
            ),
            schedule_scores=schedule_scores,
            schedule_scores_cpu=schedule_scores_cpu,
        )
