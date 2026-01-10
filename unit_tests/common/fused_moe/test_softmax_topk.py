import torch
import time
import pytest
import numpy as np
from lightllm.common.fused_moe.softmax_topk import softmax_topk
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


def benchmark(M, N, K, renorm, runs):
    try:
        import sgl_kernel as sgl_ops
    except Exception as e:
        pytest.skip(f"no sgl_kernel error: {str(e)}", allow_module_level=True)

    gating = torch.randn(M, N, device="cuda", dtype=torch.float32)
    torch.cuda.synchronize()

    # 1. SGL kernel
    sgl_vals = torch.empty((M, K), dtype=torch.float32, device="cuda")
    sgl_ids = torch.empty((M, K), dtype=torch.int32, device="cuda")
    # Warm-up
    sgl_ops.topk_softmax(sgl_vals, sgl_ids, gating)
    torch.cuda.synchronize()
    start = torch.cuda.Event(True)
    end = torch.cuda.Event(True)
    start.record()
    for _ in range(runs):
        sgl_ops.topk_softmax(sgl_vals, sgl_ids, gating)
        if renorm:
            sgl_vals.div_(sgl_vals.sum(-1, keepdim=True).clamp_min(1e-8))

    end.record()
    torch.cuda.synchronize()
    t_sgl = start.elapsed_time(end) / runs

    # 2. Triton kernel
    t0 = torch.cuda.Event(True)
    t1 = torch.cuda.Event(True)
    # Warm-up
    softmax_topk(gating, K)
    torch.cuda.synchronize()
    t0.record()
    for _ in range(runs):
        triton_vals, triton_ids = softmax_topk(gating, K, renorm)
    t1.record()
    torch.cuda.synchronize()
    t_triton = t0.elapsed_time(t1) / runs

    # 3. Native PyTorch
    _ = torch.softmax(gating, dim=-1)
    _, _ = torch.topk(_, K, dim=-1)
    torch.cuda.synchronize()

    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(runs):
        probs = torch.softmax(gating, dim=-1)
        torch_vals, torch_ids = torch.topk(probs, K, dim=-1)
        if renorm:
            torch_vals.div_(torch_vals.sum(-1, keepdim=True).clamp_min(1e-8))
    end.record()
    torch.cuda.synchronize()
    t_torch = start.elapsed_time(end) / runs

    # Compare indices and weights
    # Count mismatches of ordered indices
    diff_sgl_triton_ids = (sgl_ids != triton_ids).sum().item()
    diff_torch_triton_ids = (torch_ids != triton_ids).sum().item()
    # Max absolute difference of weights aligned by position
    max_err_triton_torch = (triton_vals - torch_vals).abs().max().item()
    max_err_triton_torch_sgl = (sgl_vals - torch_vals).abs().max().item()
    max_err_triton_sgl = (triton_vals - sgl_vals).abs().max().item()

    assert diff_sgl_triton_ids == 0, f"Mismatch SGL vs Triton ids: {diff_sgl_triton_ids}"
    assert diff_torch_triton_ids == 0, f"Mismatch Torch vs Triton ids: {diff_torch_triton_ids}"
    assert max_err_triton_torch < 1e-3, f"Max err Triton vs Torch: {max_err_triton_torch}"
    assert max_err_triton_torch_sgl < 1e-3, f"Max err Triton vs SGL: {max_err_triton_torch_sgl}"
    assert max_err_triton_sgl < 1e-3, f"Max err Torch vs SGL: {max_err_triton_sgl}"

    results = {
        "time_sgl": t_sgl,
        "time_triton": t_triton,
        "time_torch": t_torch,
        "mismatch_sgl_triton_ids": diff_sgl_triton_ids,
        "mismatch_torch_triton_ids": diff_torch_triton_ids,
        "max_err_triton_torch": max_err_triton_torch,
        "max_err_triton_sgl": max_err_triton_sgl,
        "max_err_triton_torch_sgl": max_err_triton_torch_sgl,
        "sgl_ids": sgl_ids,
        "triton_ids": triton_ids,
        "torch_ids": torch_ids,
        "sgl_vals": sgl_vals,
        "triton_vals": triton_vals,
        "torch_vals": torch_vals,
    }
    return results


def test_softmax_topk():
    M, N, K = 8192, 1024, 8
    res = benchmark(M, N, K, False, 1000)
    print(f"SGL     time: {res['time_sgl']:.6f}ms")
    print(f"Triton  time: {res['time_triton']:.6f}ms")
    print(f"PyTorch time: {res['time_torch']:.6f}ms")
    print("Mismatch SGL vs Triton ids:", res["mismatch_sgl_triton_ids"])
    print("Mismatch Torch vs Triton ids:", res["mismatch_torch_triton_ids"])
    print("Max err Triton vs Torch  :", res["max_err_triton_torch"])
    print("Max err Triton vs SGL    :", res["max_err_triton_sgl"])
    print("Max err Torch vs SGL    :", res["max_err_triton_torch_sgl"])
    benchmark(M, N, K, True, 10)
    benchmark(M, 256, 5, True, 10)
    benchmark(M, 127, 5, True, 10)


if __name__ == "__main__":
    pytest.main()
