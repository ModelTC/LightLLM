import torch


def token_decode_attention_flash_decoding(
    q,
    infer_state,
    q_head_num,
    head_dim,
    cache_k,
    cache_k_scale,
    cache_v,
    cache_v_scale,
    out=None,
    alloc_tensor_func=torch.empty,
):
    BLOCK_SEQ = 256
    batch_size = infer_state.batch_size
    max_len_in_batch = infer_state.max_len_in_batch
    calcu_shape1 = (batch_size, q_head_num, head_dim)

    from lightllm_ppl_int4kv_flashdecoding_kernel import group8_int4kv_flashdecoding_stage1
    from .flash_decoding_stage2 import flash_decode_stage2

    o_tensor = alloc_tensor_func(q.shape, q.dtype, q.device) if out is None else out

    mid_o = alloc_tensor_func(
        [batch_size, q_head_num, max_len_in_batch // BLOCK_SEQ + 1, head_dim], dtype=torch.float16, device="cuda"
    )
    mid_o_logexpsum = alloc_tensor_func(
        [batch_size, q_head_num, max_len_in_batch // BLOCK_SEQ + 1], dtype=torch.float16, device="cuda"
    )

    group8_int4kv_flashdecoding_stage1(
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
        infer_state.max_len_in_batch,
    )

    flash_decode_stage2(mid_o, mid_o_logexpsum, infer_state.b_seq_len, o_tensor.view(calcu_shape1), BLOCK_SEQ)
    return o_tensor
