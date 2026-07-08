# bench_decode.py
import os
import argparse
import torch
import triton

os.environ['PYTHONPATH'] = '/data/nvme0/chenjunyi/project/lightllm'

# 放在同目录可直接这样导

from lightllm.common.basemodel.triton_kernel.att.decode_att.gqa.mtp_diverse.mtp_diverse_attn import token_decode_attention_mtp_diverse_single_token, _prepare_split_marks
from lightllm.common.basemodel.triton_kernel.att.decode_att.gqa.mtp_diverse.stage1_single_token import mtp_diverse_stage1_single_token
from lightllm.common.basemodel.triton_kernel.att.decode_att.gqa.mtp_diverse.stage2_single_token import mtp_diverse_stage2_single_token


def seed_all(seed=2025):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_group_marks(B: int, max_group: int, device="cuda"):
    """
    构造 b_mark_shared_group: 仅组尾位置存组大小，其余为0
    """
    marks = torch.zeros((B,), dtype=torch.int32, device=device)
    i = 0
    while i < B:
        g = min(max_group, B - i)
        marks[i + g - 1] = g
        i += g
    return marks


def build_seq_lens_by_group(marks: torch.Tensor, max_kv_len: int, device="cuda"):
    """
    按组构造递增 seq_len，贴近你的注释语义
    """
    B = marks.numel()
    seq = torch.zeros((B,), dtype=torch.int32, device=device)
    start = 0
    for end in range(B):
        g = int(marks[end].item())
        if g > 0:
            base = max(2, max_kv_len // 2)
            vals = torch.linspace(base, max_kv_len, steps=g, device=device).to(torch.int32)
            seq[start:end + 1] = vals
            start = end + 1
    return seq


def make_inputs(
    B=8,
    num_reqs=8,
    q_heads=32,
    kv_heads=8,
    head_dim=128,
    max_kv_len=4096,
    total_tokens=65536,
    dtype=torch.float16,
    max_group=4,
    device="cuda",
):
    assert q_heads % kv_heads == 0
    assert num_reqs >= B, "为了简单，num_reqs 建议 >= B"

    q = torch.randn((B, q_heads, head_dim), dtype=dtype, device=device)
    k = torch.randn((total_tokens, kv_heads, head_dim), dtype=dtype, device=device)
    v = torch.randn((total_tokens, kv_heads, head_dim), dtype=dtype, device=device)

    # 每个 req 的 token 列表
    Req_to_tokens = torch.randint(
        0, total_tokens, (num_reqs, max_kv_len), dtype=torch.int32, device=device
    )
    # batch 内每个样本映射一个 req（这里直接 0..B-1）
    B_req_idx = torch.arange(B, dtype=torch.int32, device=device)

    b_mark_shared_group = build_group_marks(B, max_group=max_group, device=device)
    b_seq_len = build_seq_lens_by_group(b_mark_shared_group, max_kv_len=max_kv_len, device=device)

    return q, k, v, Req_to_tokens, B_req_idx, b_seq_len, b_mark_shared_group


@torch.no_grad()
def run_full(
    q, k, v, Req_to_tokens, B_req_idx, b_seq_len, b_mark_shared_group,
    block_seq=256, max_shared_group_size=16, stage1_run_config=None
):
    return token_decode_attention_mtp_diverse_single_token(
        q=q, k=k, v=v,
        Req_to_tokens=Req_to_tokens,
        B_req_idx=B_req_idx,
        b_seq_len=b_seq_len,
        b_mark_shared_group=b_mark_shared_group,
        block_seq=block_seq,
        max_shared_group_size=max_shared_group_size,
        stage1_run_config=stage1_run_config,
    )


@torch.no_grad()
def run_split_stage1_stage2(
    q, k, v, Req_to_tokens, B_req_idx, b_seq_len, b_mark_shared_group,
    block_seq=256, max_shared_group_size=16, stage1_run_config=None, stage2_run_config=None
):
    B, qh, d = q.shape
    max_kv_len = Req_to_tokens.shape[1]
    seq_block_num = triton.cdiv(max_kv_len, block_seq)

    mid_o = torch.empty((B, qh, seq_block_num, d), dtype=q.dtype, device=q.device)
    mid_lse = torch.empty((B, qh, seq_block_num), dtype=torch.float32, device=q.device)
    out = torch.empty_like(q)
    split_mark = torch.empty_like(b_mark_shared_group)

    # split 的 block_batch 跟 stage1 保持一致更稳
    split_block_batch = 4
    if stage1_run_config is not None and "BLOCK_BATCH" in stage1_run_config:
        split_block_batch = int(stage1_run_config["BLOCK_BATCH"])

    _prepare_split_marks(
        b_mark_shared_group=b_mark_shared_group,
        out_mark=split_mark,
        block_batch=split_block_batch,
        max_shared_group_size=max_shared_group_size,
    )

    mtp_diverse_stage1_single_token(
        q=q, k=k, v=v,
        Req_to_tokens=Req_to_tokens,
        B_req_idx=B_req_idx,
        b_seq_len=b_seq_len,
        b_mark_shared_group=split_mark,
        max_kv_len=max_kv_len,
        mid_out=mid_o,
        mid_out_logsumexp=mid_lse,
        block_seq=block_seq,
        run_config=stage1_run_config,
    )

    mtp_diverse_stage2_single_token(
        mid_out=mid_o,
        mid_out_logsumexp=mid_lse,
        B_Seqlen=b_seq_len,
        O=out,
        block_seq=block_seq,
        max_kv_len=max_kv_len,
        run_config=stage2_run_config,
    )
    return out


def bench(fn, warmup=200, iters=1000):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    st = torch.cuda.Event(enable_timing=True)
    ed = torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(iters):
        fn()
    ed.record()
    torch.cuda.synchronize()
    return st.elapsed_time(ed) / iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=8)
    parser.add_argument("--num-reqs", type=int, default=8)
    parser.add_argument("--q-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--max-kv-len", type=int, default=4096)
    parser.add_argument("--total-tokens", type=int, default=65536)
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16"])
    parser.add_argument("--block-seq", type=int, default=256)
    parser.add_argument("--max-group", type=int, default=4)
    parser.add_argument("--max-shared-group-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()

    seed_all(args.seed)
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    stage1_run_config = {
        "BLOCK_N": 32,
        "num_warps": 4,
        "num_stages": 3,
        "BLOCK_BATCH": 4,
    }
    stage2_run_config = {
        "num_warps": 4,
        "num_stages": 2,
    }

    inputs = make_inputs(
        B=args.B,
        num_reqs=args.num_reqs,
        q_heads=args.q_heads,
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        max_kv_len=args.max_kv_len,
        total_tokens=args.total_tokens,
        dtype=dtype,
        max_group=args.max_group,
        device="cuda",
    )
    q, k, v, Req_to_tokens, B_req_idx, b_seq_len, b_mark_shared_group = inputs

    # 先编译
    run_full(
        q, k, v, Req_to_tokens, B_req_idx, b_seq_len, b_mark_shared_group,
        block_seq=args.block_seq,
        max_shared_group_size=args.max_shared_group_size,
        stage1_run_config=stage1_run_config,
    )
    torch.cuda.synchronize()

    t_full = bench(
        lambda: run_full(
            q, k, v, Req_to_tokens, B_req_idx, b_seq_len, b_mark_shared_group,
            block_seq=args.block_seq,
            max_shared_group_size=args.max_shared_group_size,
            stage1_run_config=stage1_run_config,
        ),
        warmup=args.warmup,
        iters=args.iters,
    )

    t_pipe = bench(
        lambda: run_split_stage1_stage2(
            q, k, v, Req_to_tokens, B_req_idx, b_seq_len, b_mark_shared_group,
            block_seq=args.block_seq,
            max_shared_group_size=args.max_shared_group_size,
            stage1_run_config=stage1_run_config,
            stage2_run_config=stage2_run_config,
        ),
        warmup=args.warmup,
        iters=args.iters,
    )

    print("=" * 90)
    print(f"FULL API avg:    {t_full:.4f} ms")
    print(f"SPLIT+S1+S2 avg: {t_pipe:.4f} ms")
    print(f"shape: B={args.B}, qh={args.q_heads}, kvh={args.kv_heads}, D={args.head_dim}, "
          f"max_kv={args.max_kv_len}, dtype={args.dtype}, max_group={args.max_group}")
    print("=" * 90)


if __name__ == "__main__":
    main()