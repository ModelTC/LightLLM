import argparse
from dataclasses import dataclass
import random

import torch
import triton

from lightllm.common.basemodel.triton_kernel.att.decode_att.gqa.flash_decoding.gqa_flash_decoding_diverse import (
    gqa_token_decode_attention_flash_decoding_diverse,
)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class MockReqManager:
    def __init__(self, req_to_token_indexs: torch.Tensor):
        self.req_to_token_indexs = req_to_token_indexs


@dataclass
class MockInferState:
    batch_size: int
    max_kv_seq_len: int
    req_manager: MockReqManager
    b_req_idx: torch.Tensor
    b_seq_len: torch.Tensor
    b_shared_seq_len: torch.Tensor
    b_mark_shared_group: torch.Tensor


def build_case(
    batch_size: int,
    shared_prefix_len: int,
    total_seq_len: int,
    max_len_in_batch: int,
    max_batch_group_size: int,
    num_heads: int,
    kv_head_num: int,
    head_dim: int,
    dtype: torch.dtype,
):
    tail_len = total_seq_len - shared_prefix_len
    if tail_len < 0:
        raise ValueError("total_seq_len must be >= shared_prefix_len")

    seq_lens = [total_seq_len for _ in range(batch_size)]
    total_unique_tokens = shared_prefix_len + batch_size * tail_len

    q = torch.randn((batch_size, num_heads, head_dim), dtype=dtype, device="cuda")
    cache_k = torch.randn((total_unique_tokens, kv_head_num, head_dim), dtype=dtype, device="cuda")
    cache_v = torch.randn((total_unique_tokens, kv_head_num, head_dim), dtype=dtype, device="cuda")

    req_to_tokens = torch.zeros((batch_size, max_len_in_batch), dtype=torch.int32, device="cuda")
    shared_positions = torch.arange(shared_prefix_len, dtype=torch.int32, device="cuda")
    tail_offset = shared_prefix_len
    for batch_idx in range(batch_size):
        req_to_tokens[batch_idx, :shared_prefix_len] = shared_positions
        req_to_tokens[batch_idx, shared_prefix_len:total_seq_len] = torch.arange(
            tail_offset, tail_offset + tail_len, dtype=torch.int32, device="cuda"
        )
        tail_offset += tail_len

    b_seq_len = torch.full((batch_size,), total_seq_len, dtype=torch.int32, device="cuda")
    b_req_idx = torch.arange(batch_size, dtype=torch.int32, device="cuda")
    b_shared_seq_len = torch.full((batch_size,), shared_prefix_len, dtype=torch.int32, device="cuda")
    b_mark_shared_group = torch.zeros((batch_size,), dtype=torch.int32, device="cuda")

    group_start = 0
    while group_start < batch_size:
        group_size = min(max_batch_group_size, batch_size - group_start)
        group_end = group_start + group_size
        if group_size == 1:
            b_shared_seq_len[group_start] = 0
        b_mark_shared_group[group_end - 1] = group_size
        group_start = group_end

    infer_state = MockInferState(
        batch_size=batch_size,
        max_kv_seq_len=max_len_in_batch,
        req_manager=MockReqManager(req_to_tokens),
        b_req_idx=b_req_idx,
        b_seq_len=b_seq_len,
        b_shared_seq_len=b_shared_seq_len,
        b_mark_shared_group=b_mark_shared_group,
    )

    kv_indptr = torch.zeros((batch_size + 1,), dtype=torch.int32, device="cuda")
    kv_indptr[1:] = torch.cumsum(b_seq_len, dim=0)
    kv_indices = torch.empty(int(b_seq_len.sum().item()), dtype=torch.int32, device="cuda")
    write_offset = 0
    for batch_idx in range(batch_size):
        kv_indices[write_offset: write_offset + total_seq_len] = req_to_tokens[batch_idx, :total_seq_len]
        write_offset += total_seq_len
    kv_last_page_len = torch.ones((batch_size,), dtype=torch.int32, device="cuda")

    return {
        "q": q,
        "cache_k": cache_k,
        "cache_v": cache_v,
        "infer_state": infer_state,
        "kv_indptr": kv_indptr,
        "kv_indices": kv_indices,
        "kv_last_page_len": kv_last_page_len,
    }


def build_flashinfer_wrapper(case, num_heads: int, kv_head_num: int, head_dim: int):
    import flashinfer

    workspace = torch.empty(512 * 1024 * 1024, dtype=torch.int8, device="cuda")
    wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
        workspace,
        "NHD",
        use_cuda_graph=True,
        use_tensor_cores=True,
        paged_kv_indptr_buffer=case["kv_indptr"],
        paged_kv_indices_buffer=case["kv_indices"],
        paged_kv_last_page_len_buffer=case["kv_last_page_len"],
    )
    wrapper.plan(
        case["kv_indptr"],
        case["kv_indices"],
        case["kv_last_page_len"],
        num_heads,
        kv_head_num,
        head_dim,
        1,
        q_data_type=case["q"].dtype,
        kv_data_type=case["cache_k"].dtype,
        non_blocking=True,
    )
    return wrapper


def benchmark_one_case(case, num_heads: int, kv_head_num: int, head_dim: int, rep: int, max_batch_group_size: int):
    wrapper = build_flashinfer_wrapper(case, num_heads=num_heads, kv_head_num=kv_head_num, head_dim=head_dim)
    diverse_out = torch.empty_like(case["q"])
    flashinfer_out = torch.empty_like(case["q"])
    shared_streams_dict = {}

    def run_diverse():
        gqa_token_decode_attention_flash_decoding_diverse(
            q=case["q"],
            infer_state=case["infer_state"],
            cache_k=case["cache_k"],
            cache_v=case["cache_v"],
            out=diverse_out,
            shared_streams_dict=shared_streams_dict,
            max_batch_group_size=max_batch_group_size,
        )

    def run_flashinfer():
        wrapper.run(case["q"], (case["cache_k"].unsqueeze(1), case["cache_v"].unsqueeze(1)), out=flashinfer_out)

    run_diverse()
    run_flashinfer()
    torch.cuda.synchronize()

    max_diff = (diverse_out - flashinfer_out).abs().max().item()
    mean_diff = (diverse_out - flashinfer_out).abs().mean().item()
    cos_sim = torch.nn.functional.cosine_similarity(diverse_out.flatten(), flashinfer_out.flatten(), dim=0).item()

    diverse_ms = triton.testing.do_bench_cudagraph(run_diverse, rep=rep)
    flashinfer_ms = triton.testing.do_bench_cudagraph(run_flashinfer, rep=rep)

    return {
        "diverse_ms": diverse_ms,
        "flashinfer_ms": flashinfer_ms,
        "speedup": flashinfer_ms / diverse_ms,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "cos_sim": cos_sim,
    }


def resolve_total_seq_len(shape_mode: str, shared_prefix_len: int, default_total_seq_len: int, batch_size: int, seed: int) -> int:
    if shape_mode == "fixed-shape":
        return default_total_seq_len

    rng = random.Random(seed + batch_size)
    return shared_prefix_len + rng.randint(80, 120)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape-mode", choices=["fixed-shape", "old-shape"], default="fixed-shape")
    parser.add_argument("--shared-prefix-len", type=int, default=2900)
    parser.add_argument("--total-seq-len", type=int, default=3072)
    parser.add_argument("--max-len-in-batch", type=int, default=8192)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4, 10, 16, 24, 32])
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--kv-head-num", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    set_seed(args.seed)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    print(
        f"device={torch.cuda.get_device_name(0)} dtype={dtype} shape_mode={args.shape_mode} "
        f"shared_prefix_len={args.shared_prefix_len} "
        f"total_seq_len={args.total_seq_len} max_len_in_batch={args.max_len_in_batch} "
        f"group_size={args.group_size} rep={args.rep}"
    )
    print(
        f"{'batch':>5} {'group':>5} {'seq_len':>8} {'diverse_ms':>12} {'flashinfer_ms':>14} "
        f"{'speedup':>9} {'max_diff':>10} {'mean_diff':>10} {'cos_sim':>10}"
    )

    for batch_size in args.batches:
        total_seq_len = resolve_total_seq_len(
            shape_mode=args.shape_mode,
            shared_prefix_len=args.shared_prefix_len,
            default_total_seq_len=args.total_seq_len,
            batch_size=batch_size,
            seed=args.seed,
        )
        case = build_case(
            batch_size=batch_size,
            shared_prefix_len=args.shared_prefix_len,
            total_seq_len=total_seq_len,
            max_len_in_batch=args.max_len_in_batch,
            max_batch_group_size=args.group_size,
            num_heads=args.num_heads,
            kv_head_num=args.kv_head_num,
            head_dim=args.head_dim,
            dtype=dtype,
        )
        metrics = benchmark_one_case(
            case=case,
            num_heads=args.num_heads,
            kv_head_num=args.kv_head_num,
            head_dim=args.head_dim,
            rep=args.rep,
            max_batch_group_size=args.group_size,
        )
        print(
            f"{batch_size:5d} {args.group_size:5d} {total_seq_len:8d} "
            f"{metrics['diverse_ms']:12.4f} {metrics['flashinfer_ms']:14.4f} "
            f"{metrics['speedup']:9.3f} {metrics['max_diff']:10.4e} "
            f"{metrics['mean_diff']:10.4e} {metrics['cos_sim']:10.6f}"
        )


if __name__ == "__main__":
    main()
