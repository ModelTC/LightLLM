import torch
import triton
import triton.language as tl

from .stage1_single_token import mtp_diverse_stage1_single_token
from .stage2_single_token import mtp_diverse_stage2_single_token


@triton.jit
def _split_shared_group_marks_kernel(
    InMark,     # [B] int32, 原始 b_mark_shared_group
    OutMark,    # [B] int32, 拆分后的 mark（需先 zero）
    B,          # int32
    BLOCK_BATCH: tl.constexpr,
    MAX_CHUNKS: tl.constexpr,   # ceil(max_shared_group_size / BLOCK_BATCH)
):
    pid = tl.program_id(0)
    if pid >= B:
        return

    g = tl.load(InMark + pid)   # 0 或 group_size（只在组尾非0）
    is_end = g > 0
    start = pid - g + 1         # 该组起点

    for j in tl.static_range(0, MAX_CHUNKS):
        off = j * BLOCK_BATCH
        valid = is_end & (off < g)

        c = tl.minimum(BLOCK_BATCH, g - off)      # 子组大小
        sub_end = start + off + c - 1             # 子组尾位置
        valid = valid & (sub_end >= 0) & (sub_end < B)

        tl.store(OutMark + sub_end, c, mask=valid)


@torch.no_grad()
def _prepare_split_marks(
    b_mark_shared_group: torch.Tensor,
    out_mark: torch.Tensor,
    block_batch: int,
    max_shared_group_size: int,
):
    B = b_mark_shared_group.shape[0]
    out_mark.zero_()
    max_chunks = (max_shared_group_size + block_batch - 1) // block_batch

    _split_shared_group_marks_kernel[(B,)](
        b_mark_shared_group,
        out_mark,
        B,
        BLOCK_BATCH=block_batch,
        MAX_CHUNKS=max_chunks,
        num_warps=1,
        num_stages=1,
    )


@torch.no_grad()
def token_decode_attention_mtp_diverse_single_token(
    q, k, v,
    Req_to_tokens,
    B_req_idx,                 # 保持原样传给 stage1
    b_seq_len,
    b_mark_shared_group,       # 原始 mark
    block_seq: int = 256,
    max_shared_group_size: int = 16,
    out=None,
    alloc_tensor_func=torch.empty,
    split_mark_buf=None,       # 可外部分配复用
    stage1_run_config=None,    # 可选：固定 stage1 配置（含 BLOCK_BATCH）
):
    batch_size = b_seq_len.shape[0]
    num_heads = q.shape[1]
    head_dim = q.shape[2]

    if out is None:
        o_tensor = alloc_tensor_func(q.shape, dtype=q.dtype, device=q.device)
    else:
        o_tensor = out

    max_kv_len = Req_to_tokens.shape[1]
    seq_block_num = triton.cdiv(max_kv_len, block_seq)

    mid_o = alloc_tensor_func(
        [batch_size, num_heads, seq_block_num, head_dim], dtype=q.dtype, device=q.device
    )
    mid_o_logsumexp = alloc_tensor_func(
        [batch_size, num_heads, seq_block_num], dtype=torch.float32, device=q.device
    )

    if split_mark_buf is None:
        split_mark_buf = alloc_tensor_func(
            [batch_size], dtype=b_mark_shared_group.dtype, device=b_mark_shared_group.device
        )

    # 关键：split 用一个“安全最小值”，保证 <= stage1 可能的 BLOCK_BATCH
    # 若 stage1 autotune 配置是 [4, 8]，这里固定 4 最稳
    split_block_batch = 4
    if stage1_run_config is not None and "BLOCK_BATCH" in stage1_run_config:
        split_block_batch = int(stage1_run_config["BLOCK_BATCH"])

    _prepare_split_marks(
        b_mark_shared_group=b_mark_shared_group,
        out_mark=split_mark_buf,
        block_batch=split_block_batch,
        max_shared_group_size=max_shared_group_size,
    )

    mtp_diverse_stage1_single_token(
        q=q, k=k, v=v,
        Req_to_tokens=Req_to_tokens,
        B_req_idx=B_req_idx,                   # ✅ 原样
        b_seq_len=b_seq_len,
        b_mark_shared_group=split_mark_buf,    # ✅ 仅替换 mark
        max_kv_len=max_kv_len,
        mid_out=mid_o,
        mid_out_logsumexp=mid_o_logsumexp,
        block_seq=block_seq,
        run_config=stage1_run_config,
    )

    mtp_diverse_stage2_single_token(
        mid_out=mid_o,
        mid_out_logsumexp=mid_o_logsumexp,
        B_Seqlen=b_seq_len,
        O=o_tensor,
        block_seq=block_seq,
        max_kv_len=max_kv_len,
    )
    return o_tensor