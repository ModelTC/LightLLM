import torch

import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 256}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 256}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
    ],
    key=['K', 'N'],
)


@triton.jit
def dequantize_kernel(
    # Pointers to matrices
    b_ptr, b_scale_ptr, fpb_ptr,
    # Matrix dimensions
    K, N,
    stride_bk, stride_bn,
    stride_fpbk, stride_fpbn,
    # Meta-parameters
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    k_block_idx = tl.program_id(axis=0)
    n_block_idx = tl.program_id(axis=1)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    b_offs = (k_block_idx * BLOCK_SIZE_K + offs_k[:, None]) * stride_bk + \
        (n_block_idx * BLOCK_SIZE_N + offs_n[None, :]) * stride_bn
    fpb_offs = (k_block_idx * BLOCK_SIZE_K + offs_k[:, None]) * stride_fpbk + \
        (n_block_idx * BLOCK_SIZE_N + offs_n[None, :]) * stride_fpbn
    bs_offs = n_block_idx * BLOCK_SIZE_N + offs_n[None, :]
    n_mask = n_block_idx * BLOCK_SIZE_N + offs_n[None, :] < N
    mask = (k_block_idx * BLOCK_SIZE_K + offs_k[:, None] < K) & n_mask
    int_b = tl.load(b_ptr + b_offs, mask=mask, other=0.0)
    scale_b = tl.load(b_scale_ptr + bs_offs, mask=n_mask, other=0.0)
    tl.store(fpb_ptr + fpb_offs, int_b * scale_b, mask=mask)


def matmul_dequantize_int8(a, b, b_scale, out=None):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    # assert b.is_contiguous(), "Matrix B must be contiguous"
    M, K = a.shape
    K, N = b.shape
    if out == None:
        # Allocates output.
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    else:
        c = out
    fp_b = torch.empty((K, N), device=a.device, dtype=a.dtype)
    grid = lambda META: (
        triton.cdiv(K, META['BLOCK_SIZE_K']), triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    dequantize_kernel[grid](
        b, b_scale, fp_b,
        K, N,
        b.stride(0), b.stride(1),
        fp_b.stride(0), fp_b.stride(1)
    )
    torch.mm(a, fp_b, out=c)
    return c


def quantize_int8(weight, axis=0):
    # Weight shape: [H1, H2]
    # Scale shape: [H2]
    scale = weight.abs().amax(axis, keepdim=True) / 127.
    weight = (weight / scale).to(torch.int8)
    if axis == 0:
        weight = weight.t().contiguous().t()
    scale = scale.squeeze(axis)
    return weight, scale


def test_int8(M, K, N):
    import time

    print("M: {} K: {} N: {}".format(M, K, N))
    torch.manual_seed(0)
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    int_b, b_scale = quantize_int8(b)
    for _ in range(10):
        triton_output = matmul_dequantize_int8(a, int_b, b_scale.unsqueeze(0))
    torch.cuda.synchronize()
    iters = 512
    t1 = time.time()
    for _ in range(iters):
        triton_output = matmul_dequantize_int8(a, int_b, b_scale.unsqueeze(0))
    torch.cuda.synchronize()
    t2 = time.time()
    triton_time = t2 - t1
    print("Triton time cost", (t2 - t1))
    for _ in range(10):
        torch_output = torch.matmul(a, b)
    torch.cuda.synchronize()
    iters = 512
    t1 = time.time()
    for _ in range(iters):
        torch_output = torch.matmul(a, b)
    torch.cuda.synchronize()
    t2 = time.time()
    torch_time = t2 - t1
    print("Torch time cost", (t2 - t1))
    return triton_time, torch_time


def test_correct_int8(M=512, K=4096, N=4096):
    import time

    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    int_b, b_scale = quantize_int8(b)
    cos = torch.nn.CosineSimilarity(0)
    triton_output = matmul_dequantize_int8(a, int_b, b_scale)
    torch_output = torch.matmul(a, b)
    print(f"triton_output={triton_output}")        
    print(f"torch_output={torch_output}")
    print("Output cos ", cos(triton_output.flatten().to(torch.float32), torch_output.flatten().to(torch.float32)))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M', 'N', 'K'],  # Argument names to use as an x-axis for the plot
        x_vals=[32, 64, 128, 256] + [
            512 * i for i in range(1, 33)
        ],  # Different possible values for `x_name`
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot
        # Possible values for `line_arg`
        line_vals=['cublas', 'triton'],
        # Label name for the lines
        line_names=["cuBLAS", "Triton"],
        # Line styles
        styles=[('green', '-'), ('blue', '-')],
        ylabel="TFLOPS",  # Label name for the y-axis
        plot_name="matmul-performance",  # Name for the plot, used also as a file name for saving the plot.
        args={},
    )
)


def benchmark(M, N, K, provider):
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'cublas':
        a = torch.randn((M, K), device='cuda', dtype=torch.float16)
        b = torch.randn((K, N), device='cuda', dtype=torch.float16)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
    if provider == 'triton':
        a = torch.randn((M, K), device='cuda', dtype=torch.float16)
        b = torch.randn((K, N), device='cuda', dtype=torch.float16)
        intb, b_scale = quantize_int8(b)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul_dequantize_int8(a, intb, b_scale), quantiles=quantiles)
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(min_ms), perf(max_ms)


def test_model_layer(bs, sqe_len, hidden, inter, tp):
    st1 = 0
    st2 = 0
    t1, t2 = test_int8(bs * sqe_len, hidden, hidden * 3 // tp)
    st1 += t1
    st2 += t2
    t1, t2 = test_int8(bs * sqe_len, hidden // tp, hidden)
    st1 += t1
    st2 += t2
    t1, t2 = test_int8(bs * sqe_len, hidden, inter * 2 // tp)
    st1 += t1
    st2 += t2
    t1, t2 = test_int8(bs * sqe_len, inter // tp, hidden)
    st1 += t1
    st2 += t2
    print("Triton time {} Torch time {}".format(st1, st2))


if __name__ == "__main__":
    test_correct_int8()
    benchmark.run(show_plots=True, print_data=True)

    bs = 32
    hidden = 4096
    inter  = 11008
    prefill_len = 512
    decode_len = 1
    tp = 1
    test_model_layer(bs, prefill_len, hidden, inter, tp)
    test_model_layer(bs, decode_len, hidden, inter, tp)