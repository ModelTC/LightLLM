"""Parallel-block proposer 共享辅助函数。"""

from __future__ import annotations

import copy

import torch

from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
from lightllm.server.router.model_infer.mtp_speculative import utils as mtp_utils
from lightllm.server.router.model_infer.mtp_speculative.proposers.base import BaseSpecProposer


@torch.no_grad()
def fill_parallel_block_draft_model_kv_state(
    proposer: BaseSpecProposer,
    target_model_input: ModelInput,
    target_model_output: ModelOutput,
    target_next_token_ids: torch.Tensor,
) -> None:
    """使用 target hidden 初始化 parallel-block drafter 的 KV state。"""

    target_hidden = target_model_output.mtp_collector.spec_hidden
    if target_hidden.numel() == 0:
        return

    target_model_input.mtp_draft_input_hiddens = target_hidden
    proposer.backend.draft_models[0].forward(target_model_input)


def extend_parallel_block_draft_kv_cache(
    proposer: BaseSpecProposer,
    target_model_input: ModelInput,
    target_hidden: torch.Tensor,
) -> None:
    """提交本轮 target verify hidden，扩展 parallel-block drafter KV。"""

    target_model_input.total_token_num = target_model_input.batch_size
    target_model_input.prefix_total_token_num = 0
    target_model_input.is_prefill = True
    target_model_input.b_ready_cache_len = target_model_input.b_seq_len - 1
    target_model_input.b_prefill_start_loc = torch.arange(
        target_model_input.batch_size,
        dtype=torch.int32,
        device=target_hidden.device,
    )
    target_model_input.mtp_draft_input_hiddens = target_hidden
    proposer.backend.draft_models[0].forward(target_model_input)


def build_parallel_block_draft_input(
    proposer: BaseSpecProposer,
    target_model_input: ModelInput,
    target_next_token_ids: torch.Tensor,
    accepted_tail_rows: torch.Tensor,
    request_count: int,
):
    """构建 parallel-block drafter 的 accepted-token + mask-token 输入。"""

    draft_model = proposer.backend.draft_models[0]
    block_size = int(draft_model.block_size)
    extra_mem_indexes_cpu = mtp_utils.alloc_mem_indexes(request_count * block_size)

    block_input_ids = target_next_token_ids.new_full(
        (request_count * block_size,),
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
    draft_input.total_token_num = draft_input.input_ids.shape[0]
    draft_input.batch_size = draft_input.total_token_num
    draft_input.max_q_seq_len = 1
    draft_input.max_kv_seq_len = target_model_input.max_kv_seq_len + block_size
    draft_input.b_req_idx = (
        target_model_input.b_req_idx.index_select(0, accepted_tail_rows).repeat_interleave(block_size).contiguous()
    )
    draft_input.b_mtp_index = torch.zeros_like(draft_input.b_req_idx)
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
    draft_input.mem_indexes = extra_mem_indexes_cpu.to(device=target_next_token_ids.device, non_blocking=True)
    draft_input.b_mark_shared_group = torch.zeros_like(draft_input.b_req_idx)
    draft_input.b_mark_shared_group[block_size - 1 :: block_size] = block_size
    empty_multimodal_params = {"images": [], "audios": []}
    draft_input.multimodal_params = [empty_multimodal_params] * draft_input.batch_size
    return draft_input, extra_mem_indexes_cpu
