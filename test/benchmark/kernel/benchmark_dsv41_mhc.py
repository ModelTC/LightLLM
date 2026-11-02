"""Compare V4.1's separate post/shifted pre with the shared fused seam."""

import json
import statistics

import torch
from triton.testing import do_bench_cudagraph

from lightllm.common.basemodel.triton_kernel.mhc import hc_post, hc_post_pre_norm
from lightllm.common.basemodel.triton_kernel.mhc.pre_norm import hc_pre_norm_shifted


if __name__ == "__main__":
    torch.manual_seed(410)
    hidden = 5120
    for tokens in (1, 6, 16, 32, 64, 128, 8192):
        residual = torch.randn(tokens, 4 * hidden, dtype=torch.bfloat16, device="cuda")
        y = torch.randn(tokens, hidden, dtype=torch.bfloat16, device="cuda")
        mix = torch.randn(tokens, 4, 4, device="cuda").softmax(-1)
        post, previous = torch.rand(tokens, 4, device="cuda"), torch.rand(tokens, 4, device="cuda")
        fn = torch.randn(24, hidden * 4, device="cuda") * 0.005
        scale, base = torch.randn(3, device="cuda"), torch.randn(24, device="cuda")
        norm = torch.randn(hidden, dtype=torch.bfloat16, device="cuda")

        def separate():
            updated = hc_post(y, residual, mix, post, 4)
            return hc_pre_norm_shifted(updated, previous, fn, scale, base, norm, 1e-6, 1e-6, 1e-6, 20)

        def fused():
            return hc_post_pre_norm(
                y, residual, mix, post, fn, scale, base, norm, 4, 1e-6, 1e-6, 1e-6, 20, previous_pre=previous
            )

        timings = {}
        for name, func in (("separate", separate), ("fused", fused)):
            func()
            timings[name] = statistics.median(do_bench_cudagraph(func, rep=100) for _ in range(3))
        print(
            json.dumps(
                dict(tokens=tokens, hidden=hidden, milliseconds=timings, speedup=timings["separate"] / timings["fused"])
            ),
            flush=True,
        )
