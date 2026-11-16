import torch


def token_decode_attention_flash_decoding(q, batch, cache_k, cache_v, out=None, alloc_tensor_func=torch.empty):
    BLOCK_SEQ = 256
    batch_size = batch.batch_size
    max_kv_seq_len = batch.max_kv_seq_len
    q_head_num, head_dim = q.shape[1], q.shape[2]
    calcu_shape1 = (batch_size, q_head_num, head_dim)

    from .flash_decoding_stage1 import flash_decode_stage1
    from .flash_decoding_stage2 import flash_decode_stage2

    o_tensor = alloc_tensor_func(q.shape, q.dtype, q.device) if out is None else out

    mid_o = alloc_tensor_func(
        [batch_size, q_head_num, max_kv_seq_len // BLOCK_SEQ + 1, head_dim], dtype=torch.float32, device="cuda"
    )
    mid_o_logexpsum = alloc_tensor_func(
        [batch_size, q_head_num, max_kv_seq_len // BLOCK_SEQ + 1], dtype=torch.float32, device="cuda"
    )

    flash_decode_stage1(
        q.view(calcu_shape1),
        cache_k,
        cache_v,
        batch.req_to_token_indexs,
        batch.b_req_idx,
        batch.b_seq_len,
        batch.max_kv_seq_len,
        mid_o,
        mid_o_logexpsum,
        BLOCK_SEQ,
    )
    flash_decode_stage2(mid_o, mid_o_logexpsum, batch.b_seq_len, o_tensor.view(calcu_shape1), BLOCK_SEQ)
    return o_tensor
