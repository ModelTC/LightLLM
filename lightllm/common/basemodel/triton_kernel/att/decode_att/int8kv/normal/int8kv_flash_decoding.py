import torch
from typing import Optional
from lightllm.common.basemodel.infer_struct import InferStateInfo
from .int8kv_flash_decoding_stage1 import flash_decode_stage1
from .int8kv_flash_decoding_stage2 import flash_decode_stage2


@torch.no_grad()
def token_decode_attention_flash_decoding(
    q: torch.Tensor,
    infer_state: InferStateInfo,
    cache_k: torch.Tensor,
    cache_k_scale: torch.Tensor,
    cache_v: torch.Tensor,
    cache_v_scale: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    alloc_tensor_func=torch.empty,
):

    q_head_num = q.shape[1]
    head_dim = q.shape[2]

    BLOCK_SEQ = 256
    batch_size = infer_state.batch_size
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
    mid_o_logexpsum = alloc_tensor_func([batch_size, q_head_num, block_num], dtype=torch.float32, device="cuda")

    flash_decode_stage1(
        q=q.view(calcu_shape1),
        k=cache_k,
        k_scale=cache_k_scale,
        v=cache_v,
        v_scale=cache_v_scale,
        Req_to_tokens=infer_state.req_manager.req_to_token_indexs,
        B_req_idx=infer_state.b_req_idx,
        B_seq_len=infer_state.b_seq_len,
        max_len_in_batch=infer_state.max_kv_seq_len,
        mid_out=mid_o,
        mid_out_logsumexp=mid_o_logexpsum,
        block_seq=BLOCK_SEQ,
    )

    flash_decode_stage2(
        mid_out=mid_o,
        mid_out_logexpsum=mid_o_logexpsum,
        B_Seqlen=infer_state.b_seq_len,
        O=o_tensor.view(calcu_shape1),
        block_seq=BLOCK_SEQ,
    )
    return o_tensor
