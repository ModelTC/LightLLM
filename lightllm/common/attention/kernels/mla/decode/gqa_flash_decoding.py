from dataclasses import dataclass
from typing import Callable

import torch
import triton
import triton.language as tl

from .gqa_flash_decoding_config import MlaDecodeAttentionKernelConfig
from .gqa_flash_decoding_stage1 import flash_decode_stage1
from .gqa_flash_decoding_stage2 import flash_decode_stage2


@dataclass(frozen=True)
class MLADecodeSchedule:
    block_seq: torch.Tensor
    batch_start_index: torch.Tensor
    vsm_count: int
    run_config: dict[str, int]


def prepare_mla_decode_schedule(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    kv_nope: torch.Tensor,
    kv_rope: torch.Tensor,
    req_to_token_indexs: torch.Tensor,
    b_req_idx: torch.Tensor,
    b_seq_len: torch.Tensor,
    tuning_seq_len: int,
    softmax_scale: float,
    **run_config: int,
) -> MLADecodeSchedule:
    batch_size, q_head_num, kv_lora_rank = q_nope.shape
    assert q_rope.shape[-1] == 64
    if not run_config:
        run_config = MlaDecodeAttentionKernelConfig.try_to_get_best_config(
            batch_size=batch_size,
            avg_seq_len_in_batch=tuning_seq_len,
            q_head_num=q_head_num,
            q_head_dim=kv_lora_rank,
            q_rope_dim=q_rope.shape[-1],
            out_dtype=torch.bfloat16,
        )
    block_seq = torch.empty((1,), dtype=torch.int64, device=q_nope.device)
    batch_start_index = torch.empty((batch_size,), dtype=torch.int64, device=q_nope.device)
    # The occupancy probe compiles stage 1 for these strides without launching it.
    vsm_count = flash_decode_stage1(
        in_block_seq=block_seq,
        q_nope=q_nope,
        q_rope=q_rope,
        kv_nope=kv_nope,
        kv_rope=kv_rope,
        Req_to_tokens=req_to_token_indexs,
        B_req_idx=b_req_idx,
        B_Seqlen=b_seq_len,
        mid_out=torch.empty((q_head_num, 0, kv_lora_rank), dtype=torch.float32, device=q_nope.device),
        mid_out_logsumexp=torch.empty((q_head_num, 0), dtype=torch.float32, device=q_nope.device),
        softmax_scale=softmax_scale,
        get_sm_count=True,
        **run_config,
    )
    _fwd_kernel_calcu_index_and_block_seq[(1,)](
        b_seq_len,
        block_seq,
        batch_start_index,
        vsm_count,
        batch_size,
        BLOCK_N=run_config["BLOCK_N"],
        MAX_BATCH_SIZE=triton.next_power_of_2(batch_size),
        num_warps=4,
    )
    return MLADecodeSchedule(
        block_seq=block_seq, batch_start_index=batch_start_index, vsm_count=vsm_count, run_config=run_config
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
