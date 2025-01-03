import os
import torch
import torch.multiprocessing as mp
from typing import List
from lightllm.utils.log_utils import init_logger
from .gqa_flash_decoding_config import MlaDecodeAttentionKernelConfig
from lightllm.utils.device_utils import get_device_sm_count

logger = init_logger(__name__)


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
    **run_config
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

    from .gqa_flash_decoding_stage1 import flash_decode_stage1
    from .gqa_flash_decoding_stage2 import flash_decode_stage2

    o_tensor = alloc_tensor_func(q_nope.shape, q_nope.dtype, q_nope.device) if out is None else out

    mid_o_block_seq = torch.empty([1], dtype=torch.int64, device="cuda")
    mid_o_batch_start_index = alloc_tensor_func(
        [
            batch_size,
        ],
        dtype=torch.int64,
        device="cuda",
    )

    mid_o = torch.empty([q_head_num, 0, kv_lora_rank], dtype=torch.float32, device="cuda")
    mid_o_logexpsum = torch.empty([q_head_num, 0], dtype=torch.float32, device="cuda")

    vsm_count = flash_decode_stage1(
        infer_state.total_token_num_tensor,
        mid_o_block_seq,
        mid_o_batch_start_index,
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

    mid_o = torch.empty([q_head_num, vsm_count * 4 + batch_size, kv_lora_rank], dtype=torch.float32, device="cuda")
    mid_o_logexpsum = torch.empty([q_head_num, vsm_count * 4 + batch_size], dtype=torch.float32, device="cuda")

    flash_decode_stage1(
        infer_state.total_token_num_tensor,
        mid_o_block_seq,
        mid_o_batch_start_index,
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
        mid_o_block_seq,
        mid_o_batch_start_index,
        mid_o,
        mid_o_logexpsum,
        infer_state.b_seq_len,
        o_tensor.view(calcu_shape1),
        **run_config
    )
    return o_tensor
