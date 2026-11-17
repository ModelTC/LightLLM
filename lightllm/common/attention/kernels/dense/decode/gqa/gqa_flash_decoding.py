import torch


def gqa_token_decode_attention_flash_decoding(
    q: torch.Tensor,
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    req_to_token_indexs: torch.Tensor,
    b_req_idx: torch.Tensor,
    b_seq_len: torch.Tensor,
    max_len_in_batch: int,
    out=None,
    alloc_tensor_func=torch.empty,
    sliding_window=(-1, -1),
):
    batch_size = b_req_idx.shape[0]
    q_head_num, head_dim = q.shape[1], q.shape[2]
    calcu_shape1 = (batch_size, q_head_num, head_dim)

    from .gqa_flash_decoding_stage1 import flash_decode_stage1
    from .gqa_flash_decoding_stage2 import flash_decode_stage2

    o_tensor = alloc_tensor_func(q.shape, q.dtype, q.device) if out is None else out

    # Because we need to allocate some intermediate tensors, considering parallelism and
    # intermediate memory consumption, when batch_size is small, block_num is larger,
    # and when batch_size is large, block_num is smaller. This achieves a better balance
    # of memory consumption and performance.
    BLOCK_SEQ = 256
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
        v=cache_v,
        Req_to_tokens=req_to_token_indexs,
        B_req_idx=b_req_idx,
        B_Seqlen=b_seq_len,
        max_len_in_batch=max_len_in_batch,
        mid_out=mid_o,
        mid_out_logsumexp=mid_o_logexpsum,
        block_seq=BLOCK_SEQ,
        sliding_window=sliding_window,
    )
    flash_decode_stage2(
        mid_out=mid_o,
        mid_out_logexpsum=mid_o_logexpsum,
        B_Seqlen=b_seq_len,
        out=o_tensor.view(calcu_shape1),
        block_seq=BLOCK_SEQ,
        sliding_window=sliding_window,
    )
    return o_tensor
