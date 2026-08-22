import torch
from .stage1_single_token import mtp_diverse_stage1_single_token
from .stage2_single_token import mtp_diverse_stage2_single_token
from lightllm.utils.envs_utils import get_diverse_max_batch_shared_group_size


@torch.no_grad()
def token_decode_attention_mtp_diverse_single_token(
    q,
    k,
    v,
    Req_to_tokens,
    B_req_idx,
    b_seq_len,
    b_mark_shared_group,
    out=None,
    alloc_tensor_func=torch.empty,
):
    batch_size = b_seq_len.shape[0]
    num_heads = q.shape[1]
    head_dim = q.shape[2]

    if out is None:
        o_tensor = alloc_tensor_func(q.shape, dtype=q.dtype, device=q.device)
    else:
        o_tensor = out

    max_kv_len = Req_to_tokens.shape[1]

    if batch_size <= 16:
        block_num = 128
    elif batch_size <= 64:
        block_num = 64
    else:
        block_num = 32

    mid_o = alloc_tensor_func([batch_size, num_heads, block_num, head_dim], dtype=q.dtype, device=q.device)
    mid_o_logsumexp = alloc_tensor_func([batch_size, num_heads, block_num], dtype=torch.float32, device=q.device)

    BLOCK_N = mtp_diverse_stage1_single_token(
        q=q,
        k=k,
        v=v,
        Req_to_tokens=Req_to_tokens,
        B_req_idx=B_req_idx,
        b_seq_len=b_seq_len,
        b_mark_shared_group=b_mark_shared_group,
        max_kv_len=max_kv_len,
        mid_out=mid_o,
        mid_out_logsumexp=mid_o_logsumexp,
        block_batch=get_diverse_max_batch_shared_group_size(),
    )

    mtp_diverse_stage2_single_token(
        mid_out=mid_o,
        mid_out_logsumexp=mid_o_logsumexp,
        B_Seqlen=b_seq_len,
        out=o_tensor,
        block_n=BLOCK_N,
    )
    return o_tensor
