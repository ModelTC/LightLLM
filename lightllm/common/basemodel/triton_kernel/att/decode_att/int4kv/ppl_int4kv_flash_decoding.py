import torch
from typing import Optional


def token_decode_attention_flash_decoding(
    q: torch.Tensor,
    infer_state,
    cache_k: torch.Tensor,
    cache_k_scale: torch.Tensor,
    cache_v: torch.Tensor,
    cache_v_scale: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    alloc_tensor_func=torch.empty,
):
    BLOCK_SEQ = 256
    batch_size = infer_state.batch_size
    q_head_num = q.shape[1]
    head_dim = q.shape[2]
    calcu_shape1 = (batch_size, q_head_num, head_dim)

    o_tensor = alloc_tensor_func(q.shape, q.dtype, q.device) if out is None else out

    # 因为需要分配一些中间tensor，考虑到并行度和中间显存的消耗，batch_size 小的
    # 时候 block_num 较大， batch_size 大的时候 block_num 较小。这样可以达到较好
    # 的显存消耗和性能的平衡。
    if batch_size <= 16:
        block_num = 128
    elif batch_size <= 64:
        block_num = 64
    else:
        block_num = 32

    mid_o = alloc_tensor_func([batch_size, q_head_num, block_num, head_dim], dtype=q.dtype, device="cuda")
    mid_o_logexpsum = alloc_tensor_func([batch_size, q_head_num, block_num], dtype=q.dtype, device="cuda")

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
        max_kv_seq_len=infer_state.max_kv_seq_len,
        mid_out=mid_o,
        mid_out_logsumexp=mid_o_logexpsum,
        block_seq=BLOCK_SEQ,
    )

    from ..int8kv.normal.int8kv_flash_decoding_stage2 import flash_decode_stage2

    flash_decode_stage2(mid_o, mid_o_logexpsum, infer_state.b_seq_len, o_tensor.view(calcu_shape1), BLOCK_SEQ)
    return o_tensor
