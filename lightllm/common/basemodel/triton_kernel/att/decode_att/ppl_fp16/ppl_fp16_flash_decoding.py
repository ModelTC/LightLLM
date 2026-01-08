import torch
from lightllm.utils.light_utils import HAS_LIGHTLLM_KERNEL, light_ops


def token_decode_attention_flash_decoding(q, infer_state, cache_k, cache_v, out=None, alloc_tensor_func=torch.empty):
    BLOCK_SEQ = 256
    batch_size = infer_state.batch_size
    q_head_num = q.shape[1]
    head_dim = q.shape[2]
    max_kv_seq_len = infer_state.max_kv_seq_len
    calcu_shape1 = (batch_size, q_head_num, head_dim)

    from ..mha.flash_decoding.flash_decoding_stage2 import flash_decode_stage2

    o_tensor = alloc_tensor_func(q.shape, q.dtype, q.device) if out is None else out

    mid_o = alloc_tensor_func(
        [batch_size, q_head_num, max_kv_seq_len // BLOCK_SEQ + 1, head_dim], dtype=torch.float16, device="cuda"
    )
    mid_o_logexpsum = alloc_tensor_func(
        [batch_size, q_head_num, max_kv_seq_len // BLOCK_SEQ + 1], dtype=torch.float16, device="cuda"
    )

    light_ops.fp16_flashdecoding_stage1(
        BLOCK_SEQ,
        mid_o,
        mid_o_logexpsum,
        1.0 / (head_dim ** 0.5),
        q.view(calcu_shape1),
        cache_k,
        cache_v,
        infer_state.req_manager.req_to_token_indexs,
        infer_state.b_req_idx,
        infer_state.b_seq_len,
        infer_state.max_kv_seq_len,
    )

    flash_decode_stage2(mid_o, mid_o_logexpsum, infer_state.b_seq_len, o_tensor.view(calcu_shape1), BLOCK_SEQ)
    return o_tensor
