import torch


def token_decode_attention_flash_decoding(
    q,
    cache_k,
    cache_v,
    req_to_token_indexs,
    b_req_idx,
    b_seq_len,
    max_kv_seq_len,
    out=None,
    alloc_tensor_func=torch.empty,
):
    BLOCK_SEQ = 256
    batch_size = b_req_idx.shape[0]
    q_head_num, head_dim = q.shape[1], q.shape[2]
    calcu_shape1 = (batch_size, q_head_num, head_dim)

    from .flash_decoding_stage1 import flash_decode_stage1
    from .flash_decoding_stage2 import flash_decode_stage2

    o_tensor = alloc_tensor_func(q.shape, dtype=q.dtype, device=q.device) if out is None else out

    mid_o = alloc_tensor_func(
        [batch_size, q_head_num, max_kv_seq_len // BLOCK_SEQ + 1, head_dim], dtype=torch.float32, device=q.device
    )
    mid_o_logexpsum = alloc_tensor_func(
        [batch_size, q_head_num, max_kv_seq_len // BLOCK_SEQ + 1], dtype=torch.float32, device=q.device
    )

    flash_decode_stage1(
        q.view(calcu_shape1),
        cache_k,
        cache_v,
        req_to_token_indexs,
        b_req_idx,
        b_seq_len,
        max_kv_seq_len,
        mid_o,
        mid_o_logexpsum,
        BLOCK_SEQ,
    )
    flash_decode_stage2(mid_o, mid_o_logexpsum, b_seq_len, o_tensor.view(calcu_shape1), BLOCK_SEQ)
    return o_tensor
