import argparse
import statistics
import sys
from pathlib import Path

import torch

CUR_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str((CUR_DIR / "../../lightllm/common/basemodel/triton_kernel/kv_copy").resolve()))
sys.path.insert(0, str((CUR_DIR / "../../lightllm/models/deepseek3_2/triton_kernel").resolve()))

from mla_copy_kv import destindex_copy_kv
from destindex_copy_kv_flashmla_fp8 import destindex_copy_kv_flashmla_fp8


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


def _gbps(token_num: int, bytes_per_token: int, elapsed_ms: float) -> float:
    return (token_num * bytes_per_token) / (elapsed_ms / 1e3) / 1e9


def _build_random_page_mapping(token_num: int, page_size: int, device: str):
    num_pages = (token_num + page_size - 1) // page_size
    padded_token_num = num_pages * page_size
    physical_page_ids = torch.randperm(num_pages, dtype=torch.int64, device=device)
    logical_tokens = torch.arange(token_num, dtype=torch.int64, device=device)
    dest_loc = physical_page_ids[logical_tokens // page_size] * page_size + (logical_tokens % page_size)
    return dest_loc, padded_token_num


def main():
    parser = argparse.ArgumentParser(description="Benchmark DeepSeek V3.2 bf16 KV store vs fp8kv_dsa KV store")
    parser.add_argument("--tokens", type=int, default=65536)
    parser.add_argument("--tokens-list", type=int, nargs="*", default=None)
    parser.add_argument("--page-size", type=int, default=None)
    parser.add_argument("--page-size-list", type=int, nargs="*", default=[1, 64, 128, 256])
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    device = "cuda"
    dtype = torch.bfloat16
    token_list = args.tokens_list if args.tokens_list else [args.tokens]

    bf16_total_bytes = 576 * 2 + 576 * 2
    fp8_total_bytes = 576 * 2 + 656

    page_sizes = [args.page_size] if args.page_size else args.page_size_list

    print(f"iters={args.iters} warmup={args.warmup}")
    for token_num in token_list:
        kv = torch.randn((token_num, 1, 576), dtype=dtype, device=device)

        for page_size in page_sizes:
            dest_loc, page_token_num = _build_random_page_mapping(token_num, page_size, device)

            bf16_nope = torch.empty((page_token_num, 1, 512), dtype=dtype, device=device)
            bf16_rope = torch.empty((page_token_num, 1, 64), dtype=dtype, device=device)
            fp8_packed = torch.empty((page_token_num, 1, 656), dtype=torch.uint8, device=device)
            fp8_nope = fp8_packed[:, :, :512].view(torch.float8_e4m3fn)
            fp8_scale = fp8_packed[:, :, 512:528].view(torch.float32)
            fp8_rope = fp8_packed[:, :, 528:].view(torch.bfloat16)

            def run_bf16():
                destindex_copy_kv(kv[:, :, :512], kv[:, :, 512:], dest_loc, bf16_nope, bf16_rope)

            def run_fp8():
                destindex_copy_kv_flashmla_fp8(
                    kv[:, :, :512],
                    kv[:, :, 512:],
                    dest_loc,
                    fp8_nope,
                    fp8_scale,
                    fp8_rope,
                )

            bf16_ms = _time_cuda(run_bf16, warmup=args.warmup, iters=args.iters)
            fp8_ms = _time_cuda(run_fp8, warmup=args.warmup, iters=args.iters)

            print(f"page_size={page_size} seqlen={token_num}")
            print(f"bf16_kv: avg_ms={bf16_ms:.4f} total_traffic_gbps={_gbps(token_num, bf16_total_bytes, bf16_ms):.2f}")
            print(f"fp8kv_dsa: avg_ms={fp8_ms:.4f} total_traffic_gbps={_gbps(token_num, fp8_total_bytes, fp8_ms):.2f}")
            print(f"speedup={bf16_ms / fp8_ms:.3f}x compression={(576 * 2) / 656:.3f}x")


if __name__ == "__main__":
    main()
