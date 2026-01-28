import torch
from lightllm.utils.light_utils import HAS_LIGHTLLM_KERNEL, light_ops


def token_decode_attention_flash_decoding(
    q,
    infer_state,
    cache_k,
    cache_k_scale,
    cache_v,
    cache_v_scale,
    out=None,
    alloc_tensor_func=torch.empty,
):
    BLOCK_SEQ = 256
    q_head_num, head_dim = q.shape[1], q.shape[2]
    batch_size = infer_state.batch_size
    max_kv_seq_len = infer_state.max_kv_seq_len
    calcu_shape1 = (batch_size, q_head_num, head_dim)

    from ..mha.flash_decoding.flash_decoding_stage2 import flash_decode_stage2

    o_tensor = alloc_tensor_func(q.shape, q.dtype, q.device) if out is None else out

    mid_o = alloc_tensor_func(
        [batch_size, q_head_num, max_kv_seq_len // BLOCK_SEQ + 1, head_dim], dtype=q.dtype, device="cuda"
    )
    mid_o_logexpsum = alloc_tensor_func(
        [batch_size, q_head_num, max_kv_seq_len // BLOCK_SEQ + 1], dtype=q.dtype, device="cuda"
    )

    light_ops.group8_int8kv_flashdecoding_stage1(
        BLOCK_SEQ,
        mid_o,
        mid_o_logexpsum,
        1.0 / (head_dim ** 0.5),
        q.view(calcu_shape1),
        cache_k,
        cache_k_scale,
        cache_v,
        cache_v_scale,
        infer_state.req_manager.req_to_token_indexs,
        infer_state.b_req_idx,
        infer_state.b_seq_len,
        infer_state.max_kv_seq_len,
    )

    flash_decode_stage2(mid_o, mid_o_logexpsum, infer_state.b_seq_len, o_tensor.view(calcu_shape1), BLOCK_SEQ)
    return o_tensor
