import torch
import triton
import triton.language as tl

from lightllm.common.kernel_config import KernelConfigs
from frozendict import frozendict
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple
from triton import Config
from lightllm.common.triton_utils.autotuner import autotune


class Fp8ScaledMMKernelConfig(KernelConfigs):
    kernel_name: str = "fp8_scaled_mm_per_token"

    @classmethod
    @lru_cache(maxsize=200)
    def try_to_get_best_config(
        cls,
        M: int,
        N: int,
        K: int,
        out_dtype: str,
    ) -> dict:
        key_params = {
            "N": N,
            "K": K,
            "out_dtype": str(out_dtype),
        }
        key_params = frozendict(key_params)

        finded_config = cls.get_the_config(key_params)

        if finded_config:
            # find by M
            config: dict = finded_config[min(finded_config.keys(), key=lambda x: abs(int(x) - M))]
            return config
        else:
            config = {
                "BLOCK_M": 64,
                "BLOCK_N": 64,
                "BLOCK_K": 64,
                "GROUP_M": 8,
                "num_warps": 4,
                "num_stages": 3,
            }
        return config

    @classmethod
    def save_config(cls, N: int, K: int, out_dtype: str, config_json: Dict[int, Dict[int, Dict]]):

        key_params = {
            "N": N,
            "K": K,
            "out_dtype": str(out_dtype),
        }
        key_params = frozendict(key_params)

        return cls.store_config(key_params, config_json)


@triton.jit
def grouped_launch(pid, m, n, block_m: tl.constexpr, block_n: tl.constexpr, group_m: tl.constexpr):

    grid_m = tl.cdiv(m, block_m)
    grid_n = tl.cdiv(n, block_n)

    width = group_m * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * group_m, group_m)

    pid_m = group_id * group_m + (pid % group_size)
    pid_n = (pid % width) // group_size

    return pid_m, pid_n


@triton.jit
def _scaled_mm_per_token(
    A,
    B,
    out,
    Ascale,
    Bscale,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    NEED_N_MASK: tl.constexpr,
    NEED_K_MASK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_m, pid_n = grouped_launch(pid, M, N, BLOCK_M, BLOCK_N, GROUP_M)

    start_m = pid_m * BLOCK_M
    start_n = pid_n * BLOCK_N

    offs_am = start_m + tl.arange(0, BLOCK_M)
    offs_bn = start_n + tl.arange(0, BLOCK_N)

    offs_am = tl.where(offs_am < M, offs_am, 0)
    offs_bn = tl.where(offs_bn < N, offs_bn, 0)

    offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_M), BLOCK_M)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_N), BLOCK_N)

    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    Ascale_ptrs = Ascale + offs_am
    Bscale_ptrs = Bscale + offs_bn
    a_s = tl.load(Ascale_ptrs)
    b_s = tl.load(Bscale_ptrs)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        if NEED_K_MASK:
            a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        else:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    acc = acc * a_s[:, None] * b_s[None, :]

    acc = acc.to(out.dtype.element_ty)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = out + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    if NEED_N_MASK:
        mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    else:
        mask = offs_cm[:, None] < M
    tl.store(c_ptrs, acc, mask=mask)


def get_test_configs():
    fp8_gemm_configs = []

    for BLOCK_M in [8, 16, 32, 64]:
        for BLOCK_N in [64, 128, 256]:
            for BLOCK_K in [32, 64, 128, 256]:
                if BLOCK_K * BLOCK_M * BLOCK_N >= 256 * 256 * 128:
                    continue
                for num_warps in [2, 4, 8]:
                    for num_stages in [2, 3, 4, 5, 6]:
                        config = {
                            "BLOCK_M": BLOCK_M,
                            "BLOCK_N": BLOCK_N,
                            "BLOCK_K": BLOCK_K,
                            "GROUP_M": 8,
                            "num_stages": num_stages,
                            "num_warps": num_warps,
                        }
                        fp8_gemm_configs.append(config)

    return fp8_gemm_configs


def _get_static_key(A, B, out_dtype):
    M, K = A.shape
    _, N = B.shape
    return {
        "N": N,
        "K": K,
        "out_dtype": str(out_dtype),
    }


@autotune(
    kernel_name="fp8_scaled_mm_per_token:v2",
    configs_gen_func=get_test_configs,
    static_key_func=_get_static_key,
    run_key_func=lambda A: A.shape[0],
    mutates_args=["out"],
)
def fp8_scaled_mm_per_token(
    A: torch.Tensor,
    B: torch.Tensor,
    Ascale: torch.Tensor,
    Bscale: torch.Tensor,
    out_dtype: torch.dtype,
    out: torch.Tensor,
    run_config=None,
) -> torch.Tensor:
    """w8a8fp8 per-token quantization mm.

    Args:
        A: Matrix A with shape of [M, K].
        B: Matrix B with shape of [K, N].
        Ascale: per-token Quantization scale for A: [M] or [M, 1].
        Bscale: per-channel Quantization scale for B: [N] or [1, N].
        out_dtype: The data type of out.
        out: The output matrix with the shape of [M, N].
    Returns:
        torch.Tensor: out.
    """
    M, K = A.shape
    _, N = B.shape
    if not run_config:
        run_config = Fp8ScaledMMKernelConfig.try_to_get_best_config(M=M, N=N, K=K, out_dtype=out_dtype)
    NEED_N_MASK = N % run_config["BLOCK_N"] != 0
    NEED_K_MASK = K % run_config["BLOCK_K"] != 0
    grid = (triton.cdiv(M, run_config["BLOCK_M"]) * triton.cdiv(N, run_config["BLOCK_N"]),)
    _scaled_mm_per_token[grid](
        A,
        B,
        out,
        Ascale,
        Bscale,
        M,
        N,
        K,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        out.stride(0),
        out.stride(1),
        NEED_N_MASK=NEED_N_MASK,
        NEED_K_MASK=NEED_K_MASK,
        **run_config,
    )

    return out


if __name__ == "__main__":
    import time
    import os
    from lightllm.common.triton_utils.autotuner import Autotuner
    import torch.nn.functional as F

    output_dtype = torch.bfloat16
    N, K = 4096, 5120

    # 测试多个不同的 M 值
    M_list = [1, 2, 4, 8, 16, 32, 48]

    print(f"{'='*80}")
    print(f"Starting Autotune for FP8 Scaled MM (N={N}, K={K})")
    print(f"M values to test: {M_list}")
    print(f"Total configs per M: {len(get_test_configs())}")
    print(f"{'='*80}\n")

    # 准备权重矩阵 B（所有测试共享）
    B = torch.randn((N, K), dtype=output_dtype).cuda().to(torch.float8_e4m3fn).transpose(0, 1)  # [K, N]
    Bscale = torch.ones((1, N)).cuda()

    # 准备所有测试数据
    test_data = {}
    for M in M_list:
        A = torch.randn((M, K), dtype=output_dtype).cuda().to(torch.float8_e4m3fn)
        Ascale = torch.randn((M, 1)).cuda()
        out = torch.zeros((M, N), dtype=output_dtype).cuda()
        test_data[M] = {"A": A, "Ascale": Ascale, "out": out}

    # ============ Phase 0: Correctness Check ============
    print("\n" + "=" * 80)
    print("PHASE 0: Verifying Correctness Before Autotune")
    print("=" * 80)

    # 选择一个中等大小的 M 进行正确性验证
    M_verify = 16 if 16 in M_list else M_list[len(M_list) // 2]
    A_verify = test_data[M_verify]["A"]
    Ascale_verify = test_data[M_verify]["Ascale"]
    out_verify = test_data[M_verify]["out"]

    print(f"\n[Verification] Testing with M={M_verify}")

    # 计算ground truth
    d_A = A_verify.to(output_dtype) * Ascale_verify.to(output_dtype)
    d_B = B.to(output_dtype) * Bscale.to(output_dtype)
    gt_C = d_A.mm(d_B)

    # 运行kernel验证正确性
    fp8_scaled_mm_per_token(A_verify, B, Ascale_verify, Bscale, output_dtype, out_verify)

    # 计算cosine similarity
    cosine_sim = F.cosine_similarity(out_verify.flatten().unsqueeze(0), gt_C.flatten().unsqueeze(0), dim=1)
    print(f"[Verification] Cosine Similarity: {cosine_sim.item():.6f}")

    # 计算max absolute error
    max_abs_error = torch.max(torch.abs(out_verify - gt_C)).item()
    mean_abs_error = torch.mean(torch.abs(out_verify - gt_C)).item()
    print(f"[Verification] Max Absolute Error: {max_abs_error:.6e}")
    print(f"[Verification] Mean Absolute Error: {mean_abs_error:.6e}")

    # 验证阈值
    if cosine_sim.item() < 0.99:
        raise RuntimeError(f"Correctness check failed! Cosine similarity {cosine_sim.item():.6f} < 0.99")

    print("[Verification] ✅ Correctness check passed!")
    print("=" * 80)

    # ============ Phase 1: Autotune ============
    print("\n" + "=" * 80)
    print("PHASE 1: Running Autotune")
    print("=" * 80)
    Autotuner.start_autotune_warmup()

    for M in M_list:
        print(f"\n[M={M}] Running autotune...")
        A = test_data[M]["A"]
        Ascale = test_data[M]["Ascale"]
        out = test_data[M]["out"]
        fp8_scaled_mm_per_token(A, B, Ascale, Bscale, output_dtype, out)
        print(f"[M={M}] Autotune completed!")

    Autotuner.end_autotune_warmup()
    print("\n" + "=" * 80)
    print("All autotune completed! Now starting benchmarks...")
    print("=" * 80)

    # ============ Phase 2: Benchmark ============
    results = []
    from sgl_kernel import fp8_scaled_mm

    for M in M_list:
        print(f"\n{'='*80}")
        print(f"Benchmarking M={M}")
        print(f"{'='*80}")

        A = test_data[M]["A"]
        Ascale = test_data[M]["Ascale"]
        out = test_data[M]["out"]

        # 验证正确性
        print(f"[M={M}] Verifying correctness...")
        d_A = A.to(output_dtype) * Ascale.to(output_dtype)
        d_B = B.to(output_dtype) * Bscale.to(output_dtype)
        gt_C = d_A.mm(d_B)

        # 运行一次确保结果正确
        fp8_scaled_mm_per_token(A, B, Ascale, Bscale, output_dtype, out)
        sgl_res = fp8_scaled_mm(A, B, Ascale, Bscale, output_dtype)

        cosine_sim = F.cosine_similarity(out.flatten().unsqueeze(0), gt_C.flatten().unsqueeze(0), dim=1)
        sgl_cosine_sim = F.cosine_similarity(sgl_res.flatten().unsqueeze(0), gt_C.flatten().unsqueeze(0), dim=1)
        print(f"[M={M}] Cosine Similarity - Our: {cosine_sim.item():.6f}, SGL: {sgl_cosine_sim.item():.6f}")

        # Benchmark 性能
        print(f"[M={M}] Benchmarking performance...")

        # BF16 baseline
        fn_bf16 = lambda: torch.mm(d_A, d_B)
        ms_bf16 = triton.testing.do_bench(fn_bf16, warmup=25, rep=100)

        # SGL kernel
        fn_sgl = lambda: fp8_scaled_mm(A, B, Ascale, Bscale, output_dtype)
        ms_sgl = triton.testing.do_bench(fn_sgl, warmup=25, rep=100)

        # Our kernel
        fn_ours = lambda: fp8_scaled_mm_per_token(A, B, Ascale, Bscale, output_dtype, out)
        ms_ours = triton.testing.do_bench_cudagraph(fn_ours, rep=100)

        print(f"[M={M}] BF16:       {ms_bf16:.3f} ms")
        print(f"[M={M}] SGL FP8:    {ms_sgl:.3f} ms ({ms_bf16/ms_sgl:.2f}x)")
        print(f"[M={M}] Our FP8:    {ms_ours:.3f} ms ({ms_bf16/ms_ours:.2f}x)")

        results.append(
            {
                "M": M,
                "bf16_ms": ms_bf16,
                "sgl_ms": ms_sgl,
                "ours_ms": ms_ours,
                "cosine_sim": cosine_sim.item(),
            }
        )

    # 打印汇总结果
    print(f"\n{'='*80}")
    print("SUMMARY - Performance Comparison")
    print(f"{'='*80}")
    print(f"{'M':<8} {'BF16(ms)':<12} {'SGL(ms)':<12} {'Our(ms)':<12} {'vs BF16':<10} {'vs SGL':<10}")
    print(f"{'-'*80}")
    for r in results:
        vs_bf16 = f"{r['bf16_ms']/r['ours_ms']:.2f}x"
        vs_sgl = f"{r['sgl_ms']/r['ours_ms']:.2f}x"
        emoji = "🎉" if r["ours_ms"] < r["sgl_ms"] else ""
        print(
            f"{r['M']:<8} {r['bf16_ms']:<12.3f} {r['sgl_ms']:<12.3f}"
            f"{r['ours_ms']:<12.3f} {vs_bf16:<10} {vs_sgl:<10} {emoji}"
        )
    print(f"{'='*80}")
