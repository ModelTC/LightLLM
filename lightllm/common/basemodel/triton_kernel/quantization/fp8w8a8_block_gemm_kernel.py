import torch
import triton
import triton.language as tl
from lightllm.common.triton_utils.autotuner import AutotuneKernelType, autotune
from lightllm.utils.device_utils import get_device_sm_count
from typing import List


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
def _block_scaled_dot(
    A,
    B,
    Ascale,
    Bscale,
    K,
    offs_am,
    offs_bn,
    valid_rows,
    group_n,
    group_k: tl.constexpr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_Ascale_m,
    stride_Ascale_k,
    stride_Bscale_k,
    stride_Bscale_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    MASK_ROWS: tl.constexpr,
):
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    Ascale_ptrs = Ascale + offs_am * stride_Ascale_m
    Bscale_ptrs = Bscale + (offs_bn // group_n) * stride_Bscale_n

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = offs_k[None, :] < K - k * BLOCK_K
        a_mask = valid_rows[:, None] & k_mask if MASK_ROWS else k_mask
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=tl.trans(k_mask), other=0.0)
        offs_ks = k * BLOCK_K // group_k
        if MASK_ROWS:
            a_s = tl.load(Ascale_ptrs + offs_ks * stride_Ascale_k, mask=valid_rows, other=0.0)
        else:
            a_s = tl.load(Ascale_ptrs + offs_ks * stride_Ascale_k)
        b_s = tl.load(Bscale_ptrs + offs_ks * stride_Bscale_k)
        if Bscale.dtype.element_ty == tl.uint8:
            # UE8M0 exponent 0 is 2^-127, which is subnormal in FP32.
            exponent = b_s.to(tl.int32)
            bits = tl.where(exponent == 0, 0x00400000, exponent << 23)
            bits = tl.where(exponent == 255, 0x7FC00000, bits)
            b_s = bits.to(tl.float32, bitcast=True)
        # Hopper needs at least 32 for native FP8 WGMMA; zero promotes operands to FP16.
        acc += tl.dot(a, b, max_num_imprecise_acc=32 if group_k == 32 else None) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    return acc


@triton.jit
def _block_scaled_block_gemm(
    A,
    B,
    C,
    Ascale,
    Bscale,
    M,
    N,
    K,
    group_n,
    group_k: tl.constexpr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_Ascale_m,
    stride_Ascale_k,
    stride_Bscale_k,
    stride_Bscale_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid_m, pid_n = grouped_launch(tl.program_id(0), M, N, BLOCK_M, BLOCK_N, GROUP_M)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = _block_scaled_dot(
        A,
        B,
        Ascale,
        Bscale,
        K,
        rows % M,
        cols % N,
        None,
        group_n,
        group_k,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_Ascale_m,
        stride_Ascale_k,
        stride_Bscale_k,
        stride_Bscale_n,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        False,
    )
    ptrs = C + rows[:, None] * stride_cm + cols[None, :] * stride_cn
    tl.store(ptrs, acc, mask=(rows[:, None] < M) & (cols[None, :] < N))


@triton.jit
def _expanded_grouped_fp8_gemm(
    A,
    B,
    C,
    Ascale,
    Bscale,
    Ends,
    M,
    N,
    K,
    ROW_OFFSET,
    stride_as_m,
    stride_as_k,
    EXPERTS: tl.constexpr,
    EXPERT_ALIGNMENT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    expert_ids = tl.arange(0, triton.next_power_of_2(EXPERTS))
    ends = tl.load(Ends + expert_ids, expert_ids < EXPERTS, 0)
    previous_ends = tl.load(Ends + expert_ids - 1, (expert_ids > 0) & (expert_ids < EXPERTS), 0)
    starts = tl.maximum(tl.cdiv(previous_ends, EXPERT_ALIGNMENT) * EXPERT_ALIGNMENT, ROW_OFFSET)
    padded_ends = tl.minimum(tl.cdiv(ends, EXPERT_ALIGNMENT) * EXPERT_ALIGNMENT, ROW_OFFSET + M)
    tile_ends = tl.cumsum(tl.cdiv(tl.maximum(padded_ends - starts, 0), BLOCK_M))
    n_tiles = tl.cdiv(N, BLOCK_N)
    for tile in tl.range(tl.program_id(0), tl.max(tile_ends) * n_tiles, tl.num_programs(0)):
        m_tile = tile // n_tiles
        col = (tile % n_tiles) * BLOCK_N
        expert = tl.min(tl.where(m_tile < tile_ends, expert_ids, EXPERTS))
        first_tile = tl.max(tl.where(expert_ids < expert, tile_ends, 0))
        previous_end = tl.load(Ends + expert - 1, expert > 0, 0)
        start = tl.maximum(tl.cdiv(previous_end, EXPERT_ALIGNMENT) * EXPERT_ALIGNMENT, ROW_OFFSET)
        row = start - ROW_OFFSET + (m_tile - first_tile) * BLOCK_M
        end = tl.load(Ends + expert)
        padded_end = tl.cdiv(end, EXPERT_ALIGNMENT) * EXPERT_ALIGNMENT - ROW_OFFSET
        end -= ROW_OFFSET
        rows = row + tl.arange(0, BLOCK_M)
        cols = col + tl.arange(0, BLOCK_N)
        valid_rows = (rows < end) & (rows < M)
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        if row < end:
            acc = _block_scaled_dot(
                A,
                B + expert * N.to(tl.int64) * K,
                Ascale,
                Bscale + expert * N.to(tl.int64) * (K // 32),
                K,
                rows,
                cols % N,
                valid_rows,
                1,
                32,
                K,
                1,
                1,
                K,
                stride_as_m,
                stride_as_k,
                1,
                K // 32,
                BLOCK_M,
                BLOCK_N,
                32,
                True,
            )
        valid_out = (rows < M) & (rows < padded_end)
        tl.store(C + rows[:, None] * N + cols[None, :], acc, valid_out[:, None] & (cols[None, :] < N))


def _expanded_grouped_fp8_configs():
    return [
        {"BLOCK_M": m, "BLOCK_N": n, "num_warps": 4, "num_stages": stages}
        for m, n, stages in [
            (32, 64, 3),
            (32, 128, 3),
            (32, 256, 3),
            (64, 64, 3),
            (64, 128, 3),
            (64, 128, 4),
            (128, 64, 3),
        ]
    ]


def _expanded_grouped_fp8_key(activations, weights, out, expert_alignment):
    experts, n, k = weights[0].shape
    return {
        "experts": experts,
        "n": n,
        "k": k,
        "activation_scale_column_major": activations[1].stride(0) == 1,
        "expert_alignment": expert_alignment,
        "out": str(out.dtype),
    }


def _expanded_grouped_fp8_tuning_inputs(
    activations, weights, out, expert_ends, expert_alignment, row_offset=0, expected_m=None, run_config=None
):
    # Graph warmup may route every dummy token to the same experts. Benchmark
    # representative per-expert work without touching DeepEP's live buffers.
    experts, n, k = weights[0].shape
    padded_m = triton.cdiv(expected_m, expert_alignment) * expert_alignment
    rows = experts * padded_m
    device = out.device
    x = torch.zeros((rows, k), device=device, dtype=activations[0].dtype)
    if activations[1].stride(0) == 1:
        x_scale = torch.ones((k // 32, rows), device=device, dtype=activations[1].dtype).T
    else:
        x_scale = torch.ones((rows, k // 32), device=device, dtype=activations[1].dtype)
    ends = torch.arange(experts, device=device, dtype=torch.int32) * padded_m + expected_m
    output = torch.empty((rows, n), device=device, dtype=out.dtype)
    return ((x, x_scale), weights, output, ends, expert_alignment), {"expected_m": expected_m}


@autotune(
    kernel_name="expanded_grouped_fp8_gemm_1x32:v1",
    configs_gen_func=_expanded_grouped_fp8_configs,
    static_key_func=_expanded_grouped_fp8_key,
    run_key_func=lambda expected_m, out, weights: expected_m or triton.cdiv(out.shape[0], weights[0].shape[0]),
    kernel_type=AutotuneKernelType.DECODE,
    rebuild_input_func=_expanded_grouped_fp8_tuning_inputs,
    warmup_all_exist_config=False,
)
def expanded_grouped_fp8_gemm(
    activations, weights, out, expert_ends, expert_alignment, row_offset=0, expected_m=None, run_config=None
):
    """W8A8 with per-row, 32-element K groups over DeepEP's expanded layout."""
    x, x_scale = activations
    w, w_scale = weights
    experts, n, k = w.shape
    rows = x.shape[0]
    if not rows:
        return
    if run_config is None:
        expected_m = expected_m or triton.cdiv(rows, experts)
        block_m, block_n = (32, 256) if expected_m <= 32 else (64, 128)
        run_config = {"BLOCK_M": block_m, "BLOCK_N": block_n, "num_warps": 4, "num_stages": 3}
    m_tiles = triton.cdiv(rows, run_config["BLOCK_M"]) + experts - 1
    n_tiles = triton.cdiv(n, run_config["BLOCK_N"])
    programs = min(m_tiles * n_tiles, get_device_sm_count() * 2)
    _expanded_grouped_fp8_gemm[(programs,)](
        x,
        w,
        out,
        x_scale,
        w_scale,
        expert_ends,
        rows,
        n,
        k,
        row_offset,
        *x_scale.stride(),
        EXPERTS=experts,
        EXPERT_ALIGNMENT=expert_alignment,
        **run_config,
    )


def get_test_configs():
    fp8_gemm_configs = [
        {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8, "num_stages": 3, "num_warps": 8},
        {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 8, "num_stages": 4, "num_warps": 4},
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8, "num_stages": 4, "num_warps": 4},
        {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8, "num_stages": 4, "num_warps": 4},
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8, "num_stages": 4, "num_warps": 4},
        {"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_M": 8, "num_stages": 4, "num_warps": 4},
        {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_M": 8, "num_stages": 5, "num_warps": 2},
        {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8, "num_stages": 5, "num_warps": 2},
        {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_M": 8, "num_stages": 3, "num_warps": 8},
        {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8, "num_stages": 3, "num_warps": 8},
        {"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 128, "GROUP_M": 8, "num_stages": 4, "num_warps": 4},
        {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_M": 8, "num_stages": 4, "num_warps": 4},
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8, "num_stages": 4, "num_warps": 4},
        {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8, "num_stages": 4, "num_warps": 4},
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8, "num_stages": 4, "num_warps": 4},
        {"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 64, "GROUP_M": 8, "num_stages": 4, "num_warps": 4},
    ]
    return fp8_gemm_configs


def _get_static_key(A, B, block_size, dtype):
    M, K = A.shape
    _, N = B.shape
    return {
        "N": N,
        "K": K,
        "block_size": block_size,
        "out_dtype": str(dtype),
    }


@autotune(
    kernel_name="w8a8_block_fp8_matmul:v1",
    configs_gen_func=get_test_configs,
    static_key_func=_get_static_key,
    run_key_func=lambda A: A.shape[0],
    mutates_args=["C"],
)
def w8a8_block_fp8_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    Ascale: torch.Tensor,
    Bscale: torch.Tensor,
    C: torch.Tensor,
    block_size: List[int],
    dtype: torch.dtype = torch.bfloat16,
    run_config=None,
) -> torch.Tensor:
    """w8a8fp8 block-wise quantization mm.

    Args:
        A: Matrix A with shape of [M, K].
        B: Matrix B with shape of [K, N].
        Ascale: per-token block-wise Quantization scale for A: [M, K / block_size[0]].
        Bscale: Quantization scale for B: [K / block_size[0], M / block_size[1]].
        C: The output matrix with the shape of [M, N].
        block_size: block granularity of quantization  (e.g., [128, 128]).
        dtype: The data type of C.
    Returns:
        torch.Tensor: C.
    """
    assert len(block_size) == 2
    block_k, block_n = block_size[0], block_size[1]
    assert A.shape[0] == Ascale.shape[0] and A.shape[-1] == B.shape[0]
    assert A.is_contiguous() and C.is_contiguous()
    M, K = A.shape
    _, N = B.shape
    assert triton.cdiv(K, block_k) == Ascale.shape[-1] and Ascale.shape[-1] == Bscale.shape[0]
    assert triton.cdiv(N, block_n) == Bscale.shape[1]
    if not run_config:
        run_config = {
            "BLOCK_M": 64,
            "BLOCK_N": block_size[0],
            "BLOCK_K": block_size[1],
            "GROUP_M": 32,
            "num_warps": 4,
            "num_stages": 3,
        }
    # A dot tile must not cross a K block with a different quantization scale.
    run_config = {**run_config, "BLOCK_K": min(run_config["BLOCK_K"], block_k)}
    grid = (triton.cdiv(M, run_config["BLOCK_M"]) * triton.cdiv(N, run_config["BLOCK_N"]),)
    _block_scaled_block_gemm[grid](
        A,
        B,
        C,
        Ascale,
        Bscale,
        M,
        N,
        K,
        block_n,
        block_k,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        C.stride(0),
        C.stride(1),
        Ascale.stride(0),
        Ascale.stride(1),
        Bscale.stride(0),
        Bscale.stride(1),
        **run_config,
    )

    return C


if __name__ == "__main__":
    import time

    block_size = 128
    output_dtype = torch.bfloat16
    M, N, K = 4096, 256, 7168
    A = torch.randn((M, K), dtype=output_dtype).cuda().to(torch.float8_e4m3fn)  # Activation
    B = torch.randn((K, N), dtype=output_dtype).cuda().to(torch.float8_e4m3fn)  # Weight
    Ascale = torch.randn((M, K // block_size)).cuda()  # + 0.2
    Bscale = torch.ones((K // block_size, N // block_size)).cuda()

    C = torch.randn((M, N), dtype=output_dtype).cuda()  # weight
    B = B.T.contiguous().T
    # warmup

    w8a8_block_fp8_matmul(A, B, Ascale, Bscale, C, (block_size, block_size), output_dtype)

    #### groud truth
    print(Ascale.unsqueeze(-1).repeat(1, 1, block_size).reshape(M, K).to(output_dtype).shape)
    d_A = A.to(output_dtype) * (Ascale.unsqueeze(-1).repeat(1, 1, block_size).reshape(M, K).to(output_dtype))
    d_B = B.to(output_dtype).contiguous()

    gt_C = d_A.mm(d_B)
    # caluate the simlarity
    import torch.nn.functional as F

    cosine_sim = F.cosine_similarity(C.flatten().unsqueeze(0), gt_C.flatten().unsqueeze(0), dim=1)

    print(f"Cosine Similarity between C and gt_C: {cosine_sim.item()}")

    fn2 = lambda: torch.mm(d_A, d_B, out=gt_C)
    ms2 = triton.testing.do_bench(fn2)
    print(f"bf16 time : {ms2} ms")

    fn2 = lambda: w8a8_block_fp8_matmul(A, B, Ascale, Bscale, C, (block_size, block_size), output_dtype)
    ms2 = triton.testing.do_bench_cudagraph(fn2)
    print(f"fp8 time : {ms2} ms")
