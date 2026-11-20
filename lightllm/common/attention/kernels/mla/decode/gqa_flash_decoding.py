from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import torch
import triton
import triton.language as tl

from .gqa_flash_decoding_config import MlaDecodeAttentionKernelConfig
from .gqa_flash_decoding_stage1 import flash_decode_stage1, flash_decode_stage1_vsm_count
from .gqa_flash_decoding_stage2 import flash_decode_stage2


@dataclass(frozen=True)
class MLADecodeSchedule:
    """Split plan shared by every layer of one decode batch.

    block_seq and batch_start_index are device buffers that update_mla_decode_schedule
    rewrites in place from the current lengths; vsm_count and run_config stay fixed.
    """

    block_seq: torch.Tensor
    batch_start_index: torch.Tensor
    vsm_count: int
    run_config: dict[str, int]


@lru_cache(maxsize=256)
def _vsm_count(
    batch_size: int,
    q_head_num: int,
    kv_lora_rank: int,
    q_rope_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    run_config_items: tuple[tuple[str, int], ...],
) -> int:
    return flash_decode_stage1_vsm_count(
        batch_size, q_head_num, kv_lora_rank, q_rope_dim, dtype, device, **dict(run_config_items)
    )


def plan_mla_decode_schedule(
    b_seq_len: torch.Tensor,
    q_head_num: int,
    kv_lora_rank: int,
    q_rope_dim: int,
    dtype: torch.dtype,
    tuning_seq_len: int,
    **run_config: int,
) -> MLADecodeSchedule:
    """Allocate the schedule buffers and compute splits for the current b_seq_len."""
    batch_size = b_seq_len.shape[0]
    if not run_config:
        run_config = MlaDecodeAttentionKernelConfig.try_to_get_best_config(
            batch_size=batch_size,
            avg_seq_len_in_batch=tuning_seq_len,
            q_head_num=q_head_num,
            q_head_dim=kv_lora_rank,
            q_rope_dim=q_rope_dim,
            out_dtype=torch.bfloat16,
        )
    vsm_count = _vsm_count(
        batch_size, q_head_num, kv_lora_rank, q_rope_dim, dtype, b_seq_len.device, tuple(sorted(run_config.items()))
    )
    schedule = MLADecodeSchedule(
        block_seq=torch.empty((1,), dtype=torch.int64, device=b_seq_len.device),
        batch_start_index=torch.empty((batch_size,), dtype=torch.int64, device=b_seq_len.device),
        vsm_count=vsm_count,
        run_config=dict(run_config),
    )
    update_mla_decode_schedule(schedule, b_seq_len)
    return schedule


def update_mla_decode_schedule(schedule: MLADecodeSchedule, b_seq_len: torch.Tensor) -> None:
    """Recompute splits in place from b_seq_len, keeping the buffer addresses and grid size."""
    batch_size = schedule.batch_start_index.shape[0]
    assert b_seq_len.shape[0] == batch_size
    _fwd_kernel_calcu_index_and_block_seq[(1,)](
        b_seq_len,
        schedule.block_seq,
        schedule.batch_start_index,
        schedule.vsm_count,
        batch_size,
        BLOCK_N=schedule.run_config["BLOCK_N"],
        MAX_BATCH_SIZE=triton.next_power_of_2(batch_size),
        num_warps=4,
    )


def gqa_token_decode_attention_flash_decoding(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    kv_nope: torch.Tensor,
    kv_rope: torch.Tensor,
    req_to_token_indexs: torch.Tensor,
    b_req_idx: torch.Tensor,
    b_seq_len: torch.Tensor,
    softmax_scale: float,
    schedule: MLADecodeSchedule,
    out: torch.Tensor,
    alloc_tensor_func: Callable[..., torch.Tensor] = torch.empty,
) -> torch.Tensor:
    batch_size, q_head_num, kv_lora_rank = q_nope.shape
    slots = schedule.vsm_count * 4 + batch_size
    mid_o = alloc_tensor_func((q_head_num, slots, kv_lora_rank), dtype=torch.float32, device=q_nope.device)
    mid_o_logexpsum = alloc_tensor_func((q_head_num, slots), dtype=torch.float32, device=q_nope.device)
    flash_decode_stage1(
        in_block_seq=schedule.block_seq,
        q_nope=q_nope,
        q_rope=q_rope,
        kv_nope=kv_nope,
        kv_rope=kv_rope,
        Req_to_tokens=req_to_token_indexs,
        B_req_idx=b_req_idx,
        B_Seqlen=b_seq_len,
        mid_out=mid_o,
        mid_out_logsumexp=mid_o_logexpsum,
        softmax_scale=softmax_scale,
        num_sm=schedule.vsm_count,
        **schedule.run_config,
    )
    flash_decode_stage2(
        out_block_seq=schedule.block_seq,
        batch_start_index=schedule.batch_start_index,
        mid_out=mid_o,
        mid_out_logexpsum=mid_o_logexpsum,
        B_Seqlen=b_seq_len,
        Out=out,
        **schedule.run_config,
    )
    return out


@triton.jit
def _fwd_kernel_calcu_index_and_block_seq(
    b_seq_len_ptr,
    mid_o_decode_att_block_seq_ptr,
    mid_o_batch_start_index_ptr,
    num_sm,
    batch_size,
    BLOCK_N: tl.constexpr,
    MAX_BATCH_SIZE: tl.constexpr,
):
    b_seq_len = tl.load(
        b_seq_len_ptr + tl.arange(0, MAX_BATCH_SIZE), mask=tl.arange(0, MAX_BATCH_SIZE) < batch_size, other=0
    )
    total_token_num = tl.sum(b_seq_len)

    block_seq = tl.cast(total_token_num / (num_sm * 4), dtype=tl.int32) + 1
    block_seq = tl.cdiv(block_seq, BLOCK_N) * BLOCK_N

    block_seq_len = tl.cdiv(b_seq_len, block_seq)
    cumsum_seq_len = tl.cumsum(block_seq_len)
    batch_start_index = cumsum_seq_len - block_seq_len
    tl.store(
        mid_o_batch_start_index_ptr + tl.arange(0, MAX_BATCH_SIZE),
        batch_start_index,
        mask=tl.arange(0, MAX_BATCH_SIZE) < batch_size,
    )
    tl.store(mid_o_decode_att_block_seq_ptr, block_seq)
    return
