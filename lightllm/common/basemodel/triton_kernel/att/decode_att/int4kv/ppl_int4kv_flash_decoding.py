import torch


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
    batch_size = infer_state.batch_size
    max_kv_seq_len = infer_state.max_kv_seq_len
    q_head_num = q.shape[1]
    head_dim = q.shape[2]
    calcu_shape1 = (batch_size, q_head_num, head_dim)

    from ..mha.flash_decoding.flash_decoding_stage2 import flash_decode_stage2

    o_tensor = alloc_tensor_func(q.shape, q.dtype, q.device) if out is None else out

    mid_o = alloc_tensor_func(
        [batch_size, q_head_num, max_kv_seq_len // BLOCK_SEQ + 1, head_dim], dtype=q.dtype, device="cuda"
    )
    mid_o_logexpsum = alloc_tensor_func(
        [batch_size, q_head_num, max_kv_seq_len // BLOCK_SEQ + 1], dtype=q.dtype, device="cuda"
    )

    from .int4kv_flash_decoding_stage1 import int4kv_flash_decode_stage1

    int4kv_flash_decode_stage1(
        q=q.view(calcu_shape1),
        k=cache_k,
        k_scale=cache_k_scale,
        v=cache_v,
        v_scale=cache_v_scale,
        Req_to_tokens=infer_state.req_manager.req_to_token_indexs,
        B_req_idx=infer_state.b_req_idx,
        B_Seqlen=infer_state.b_seq_len,
        max_len_in_batch=infer_state.max_kv_seq_len,
        mid_out=mid_o,
        mid_out_logsumexp=mid_o_logexpsum,
        block_seq=BLOCK_SEQ,
    )

    flash_decode_stage2(mid_o, mid_o_logexpsum, infer_state.b_seq_len, o_tensor.view(calcu_shape1), BLOCK_SEQ)
    return o_tensor
