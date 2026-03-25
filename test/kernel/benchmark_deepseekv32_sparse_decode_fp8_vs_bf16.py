import argparse
import statistics
import sys
from pathlib import Path

import torch

CUR_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str((CUR_DIR / "../../lightllm/models/deepseek3_2/triton_kernel").resolve()))
sys.path.insert(0, str((CUR_DIR / "../../lightllm/utils").resolve()))

from destindex_copy_kv_flashmla_fp8 import pack_kv_reference
from flashmla_utils import import_flash_mla


def _time_cuda(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples_ms = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples_ms.append(start.elapsed_time(end))
    return statistics.mean(samples_ms)


def _gbps(batch: int, topk: int, bytes_per_token: int, elapsed_ms: float) -> float:
    return (batch * topk * bytes_per_token) / (elapsed_ms / 1e3) / 1e9


def _build_random_page_layout(
    cache_tokens: int,
    page_size: int,
    device: str,
    dtype: torch.dtype,
):
    num_pages = (cache_tokens + page_size - 1) // page_size
    padded_cache_tokens = num_pages * page_size
    physical_page_ids = torch.randperm(num_pages, dtype=torch.int64, device=device)

    logical_kv = torch.randn((cache_tokens, 1, 576), dtype=dtype, device=device)
    physical_kv = torch.zeros((padded_cache_tokens, 1, 576), dtype=dtype, device=device)

    logical_tokens = torch.arange(cache_tokens, dtype=torch.int64, device=device)
    physical_token_locs = physical_page_ids[logical_tokens // page_size] * page_size + (logical_tokens % page_size)
    physical_kv[physical_token_locs] = logical_kv

    return physical_kv, physical_page_ids, physical_token_locs, padded_cache_tokens


def main():
    parser = argparse.ArgumentParser(description="Benchmark DeepSeek V3.2 decode: bf16 sparse-selected vs fp8 DSA")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--heads", type=int, default=128)
    parser.add_argument("--cache-tokens", type=int, default=131072)
    parser.add_argument("--cache-tokens-list", type=int, nargs="*", default=None)
    parser.add_argument("--page-size", type=int, default=None)
    parser.add_argument("--page-size-list", type=int, nargs="*", default=[1, 64, 128, 256])
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--check-correctness", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    from sgl_kernel.flash_attn import flash_attn_with_kvcache

    flash_mla = import_flash_mla()
    device = "cuda"
    dtype = torch.bfloat16
    sm_scale = 576 ** (-0.5)
    cache_token_list = args.cache_tokens_list if args.cache_tokens_list else [args.cache_tokens]
    page_sizes = [args.page_size] if args.page_size else args.page_size_list

    q_nope = torch.randn((args.batch, args.heads, 512), dtype=dtype, device=device)
    q_rope = torch.randn((args.batch, args.heads, 64), dtype=dtype, device=device)
    q_all = torch.cat([q_nope, q_rope], dim=-1).unsqueeze(1).contiguous()
    print(f"batch={args.batch} heads={args.heads} topk={args.topk} iters={args.iters} warmup={args.warmup}")
    for idx, cache_tokens in enumerate(cache_token_list):
        for page_idx, page_size in enumerate(page_sizes):
            physical_kv, physical_page_ids, physical_token_locs, padded_cache_tokens = _build_random_page_layout(
                cache_tokens, page_size, device, dtype
            )
            num_pages = padded_cache_tokens // page_size

            k_rope = physical_kv[:, :, 512:].view(num_pages, page_size, 1, 64).contiguous()
            kv_nope = physical_kv[:, :, :512].view(num_pages, page_size, 1, 512).contiguous()
            kv_fp8 = pack_kv_reference(physical_kv).view(num_pages, page_size, 1, 656).contiguous()

            selected_pages = (args.topk + page_size - 1) // page_size
            page_table = physical_page_ids[:selected_pages].to(torch.int32).repeat(args.batch, 1)
            cache_seqlens = torch.full((args.batch,), args.topk, dtype=torch.int32, device=device)
            cu_seqlens_q = torch.arange(0, args.batch + 1, dtype=torch.int32, device=device)
            cu_seqlens_k_new = torch.arange(
                0, (args.batch + 1) * args.topk, args.topk, dtype=torch.int32, device=device
            )
            fp8_indices = (
                physical_token_locs[: args.topk].to(torch.int32).view(1, 1, args.topk).repeat(args.batch, 1, 1)
            )
            sched_meta, _ = flash_mla.get_mla_metadata()

            def run_bf16():
                return flash_attn_with_kvcache(
                    q=q_rope,
                    k_cache=k_rope,
                    v_cache=kv_nope,
                    qv=q_nope,
                    page_table=page_table,
                    cache_seqlens=cache_seqlens,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k_new=cu_seqlens_k_new,
                    max_seqlen_q=1,
                    softmax_scale=sm_scale,
                    causal=True,
                )

            def run_fp8():
                out, _ = flash_mla.flash_mla_with_kvcache(
                    q=q_all,
                    k_cache=kv_fp8,
                    block_table=None,
                    cache_seqlens=None,
                    head_dim_v=512,
                    tile_scheduler_metadata=sched_meta,
                    num_splits=None,
                    softmax_scale=sm_scale,
                    causal=False,
                    is_fp8_kvcache=True,
                    indices=fp8_indices,
                )
                return out[:, 0]

            if args.check_correctness and idx == 0 and page_idx == 0:
                bf16_out = run_bf16()
                fp8_out = run_fp8()
                max_diff = (bf16_out - fp8_out).abs().max().item()
                mean_diff = (bf16_out - fp8_out).abs().mean().item()
                print(f"correctness: max_diff={max_diff:.6f} mean_diff={mean_diff:.6f}")

            bf16_ms = _time_cuda(run_bf16, warmup=args.warmup, iters=args.iters)
            fp8_ms = _time_cuda(run_fp8, warmup=args.warmup, iters=args.iters)

            print(f"page_size={page_size} seqlen={cache_tokens}")
            print(
                f"bf16_decode: avg_ms={bf16_ms:.4f} kv_read_gbps={_gbps(args.batch, args.topk, 576 * 2, bf16_ms):.2f}"
            )
            print(f"fp8_dsa_decode: avg_ms={fp8_ms:.4f} kv_read_gbps={_gbps(args.batch, args.topk, 656, fp8_ms):.2f}")
            print(f"speedup={bf16_ms / fp8_ms:.3f}x")


if __name__ == "__main__":
    main()
