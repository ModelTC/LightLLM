import os
import torch
import torch.multiprocessing as mp
import triton
import triton.language as tl
from typing import List
from lightllm.utils.log_utils import init_logger
from .gqa_flash_decoding_config import MlaDecodeAttentionKernelConfig
from lightllm.utils.device_utils import get_device_sm_count
from lightllm.common.triton_utils.autotuner import autotune, nearest_power_of_2

logger = init_logger(__name__)
MAX_BATCH_SIZE = 1000000

def get_test_configs():
    configs = []
    for block_n in [16, 32]:
        for block_q_head in [16,32]:
            for stage1_num_warps in [2, 4, 8, 16]:
                for stage1_num_stages in [
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    12,
                    15,
                ]:
                    for stage2_num_warps in [1, 2, 4]:
                        for stage2_num_stages in [
                            1,
                            3,
                        ]:
                            t_config = {
                                "BLOCK_N": block_n,
                                "BLOCK_Q_HEAD": block_q_head,
                                "stage1_num_warps": stage1_num_warps,
                                "stage1_num_stages": stage1_num_stages,
                                "stage2_num_warps": stage2_num_warps,
                                "stage2_num_stages": stage2_num_stages,
                            }
                            configs.append(t_config)
    return configs

def get_static_key(q_head_num, kv_lora_rank, q_rope_dim):
    return {
        "q_head_num": q_head_num,
        "q_head_dim": kv_lora_rank,
        "q_rope_dim": q_rope_dim,
        "out_dtype": str(torch.bfloat16),
    }

def get_run_key(infer_state):
    if torch.cuda.is_current_stream_capturing():
        avg_seq_len_in_batch = infer_state.max_len_in_batch
    else:
        avg_seq_len_in_batch = infer_state.total_token_num // infer_state.batch_size
    return str(nearest_power_of_2(avg_seq_len_in_batch)*MAX_BATCH_SIZE + nearest_power_of_2(infer_state.batch_size))

@autotune(
    name="gqa_token_decode_attention_flash_decoding:v1",
    configs=get_test_configs(),
    default_config={
        "BLOCK_N": 16,
        "BLOCK_Q_HEAD": 16,
        "stage1_num_warps": 4,
        "stage1_num_stages": 2,
        "stage2_num_warps": 4,
        "stage2_num_stages": 2,
    },
    static_key_func=get_static_key,
    run_key_func=get_run_key,
)
def gqa_token_decode_attention_flash_decoding(
    q_nope,
    q_rope,
    kv_nope,
    kv_rope,
    infer_state,
    q_head_num,
    kv_lora_rank,
    q_rope_dim,
    qk_nope_head_dim,
    softmax_scale,
    out=None,
    alloc_tensor_func=torch.empty,
    run_config=None,
):
    batch_size = infer_state.batch_size
    max_len_in_batch = infer_state.max_len_in_batch
    calcu_shape1 = (batch_size, q_head_num, kv_lora_rank)
    calcu_shape2 = (batch_size, q_head_num, q_rope_dim)

    if not run_config:
        if torch.cuda.is_current_stream_capturing():
            avg_seq_len_in_batch = max_len_in_batch
        else:
            avg_seq_len_in_batch = infer_state.total_token_num // batch_size

        run_config = MlaDecodeAttentionKernelConfig.try_to_get_best_config(
            batch_size=batch_size,
            avg_seq_len_in_batch=avg_seq_len_in_batch,
            q_head_num=q_head_num,
            q_head_dim=kv_lora_rank,
            q_rope_dim=q_rope_dim,
            out_dtype=torch.bfloat16,
        )

    BLOCK_N = run_config["BLOCK_N"]

    from .gqa_flash_decoding_stage1 import flash_decode_stage1
    from .gqa_flash_decoding_stage2 import flash_decode_stage2

    o_tensor = alloc_tensor_func(q_nope.shape, q_nope.dtype, q_nope.device) if out is None else out

    fake_decode_att_block_seq = torch.empty([0], dtype=torch.int64, device="cuda")
    mid_o = torch.empty([q_head_num, 0, kv_lora_rank], dtype=torch.float32, device="cuda")
    mid_o_logexpsum = torch.empty([q_head_num, 0], dtype=torch.float32, device="cuda")

    vsm_count = flash_decode_stage1(
        fake_decode_att_block_seq,
        q_nope.view(calcu_shape1),
        q_rope.view(calcu_shape2),
        kv_nope,
        kv_rope,
        infer_state.req_manager.req_to_token_indexs,
        infer_state.b_req_idx,
        infer_state.b_seq_len,
        mid_o,
        mid_o_logexpsum,
        softmax_scale,
        get_sm_count=True,
        **run_config
    )

    if not hasattr(infer_state, "decode_att_block_seq"):
        assert batch_size <= 2048
        decode_att_block_seq = torch.empty(
            [
                1,
            ],
            dtype=torch.int64,
            device="cuda",
        )
        mid_o_batch_start_index = torch.empty(
            [
                batch_size,
            ],
            dtype=torch.int64,
            device="cuda",
        )
        _fwd_kernel_calcu_index_and_block_seq[(1,)](
            infer_state.b_seq_len,
            decode_att_block_seq,
            mid_o_batch_start_index,
            vsm_count,
            batch_size,
            BLOCK_N=BLOCK_N,
            num_warps=4,
        )

        infer_state.decode_att_block_seq = decode_att_block_seq
        infer_state.mid_o_batch_start_index = mid_o_batch_start_index

    mid_o = torch.empty([q_head_num, vsm_count * 4 + batch_size, kv_lora_rank], dtype=torch.float32, device="cuda")
    mid_o_logexpsum = torch.empty([q_head_num, vsm_count * 4 + batch_size], dtype=torch.float32, device="cuda")

    flash_decode_stage1(
        infer_state.decode_att_block_seq,
        q_nope.view(calcu_shape1),
        q_rope.view(calcu_shape2),
        kv_nope,
        kv_rope,
        infer_state.req_manager.req_to_token_indexs,
        infer_state.b_req_idx,
        infer_state.b_seq_len,
        mid_o,
        mid_o_logexpsum,
        softmax_scale,
        get_sm_count=False,
        **run_config
    )

    flash_decode_stage2(
        infer_state.decode_att_block_seq,
        infer_state.mid_o_batch_start_index,
        mid_o,
        mid_o_logexpsum,
        infer_state.b_seq_len,
        o_tensor.view(calcu_shape1),
        **run_config
    )
    return o_tensor


@triton.jit
def _fwd_kernel_calcu_index_and_block_seq(
    b_seq_len_ptr,
    mid_o_decode_att_block_seq_ptr,
    mid_o_batch_start_index_ptr,
    num_sm,
    batch_size,
    BLOCK_N: tl.constexpr,
):
    b_seq_len = tl.load(b_seq_len_ptr + tl.arange(0, 2048), mask=tl.arange(0, 2048) < batch_size, other=0)
    total_token_num = tl.sum(b_seq_len)

    block_seq = tl.cast(total_token_num / (num_sm * 4), dtype=tl.int32) + 1
    block_seq = tl.cdiv(block_seq, BLOCK_N) * BLOCK_N

    block_seq_len = tl.cdiv(b_seq_len, block_seq)
    cumsum_seq_len = tl.cumsum(block_seq_len)
    batch_start_index = cumsum_seq_len - block_seq_len
    tl.store(mid_o_batch_start_index_ptr + tl.arange(0, 2048), batch_start_index, mask=tl.arange(0, 2048) < batch_size)
    tl.store(mid_o_decode_att_block_seq_ptr, block_seq)
    return
