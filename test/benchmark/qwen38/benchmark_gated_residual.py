import argparse
import json
import os

import torch
import torch.nn.functional as F
import triton
from triton.testing import do_bench_cudagraph

from lightllm.common.gated_residual import GatedResidual


def eager(component, residual, update, precise_projections=False):
    def project(inputs, weight):
        if precise_projections:
            return F.linear(inputs.double(), weight.double()).to(inputs.dtype)
        return F.linear(inputs, weight)

    grouped = residual.float().unflatten(-1, (component.branches, -1))
    normalized = grouped * torch.rsqrt(grouped.square().mean(-1, keepdim=True) + component.eps)
    normalized = (normalized.flatten(-2) * (1 + component.norm_weight.float())).to(residual.dtype)
    projected = F.silu(project(normalized, component.down_weight) / component.branches)
    mix = torch.sigmoid(project(projected, component.up_weight))
    mixed = (normalized * mix).unflatten(-1, (component.branches, -1)).mean(-2)
    gates = 2 * torch.sigmoid(project(normalized, component.inject_weight) / component.branches)
    return mixed, residual + (update.unsqueeze(-2) * gates.unsqueeze(-1)).flatten(-2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 16, 128, 1024])
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if not os.environ.get("EXP_ACTIVE"):
        parser.error('Wrap this benchmark with exp -m "purpose"')
    torch.manual_seed(37)
    dim, branches, rank = 2560, 4, 320
    width = dim * branches

    def rand(*shape):
        return torch.randn(*shape, device="cuda", dtype=torch.bfloat16)

    component = GatedResidual(
        rand(width) * 0.1,
        rand(rank, width) / width ** 0.5,
        rand(width, rank) / rank ** 0.5,
        rand(branches, width) / width ** 0.5,
        branches,
    )
    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "triton": triton.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(),
                "dtype": "bfloat16",
                "hidden_size": dim,
                "branches": branches,
                "rank": rank,
                "timing": "CUDA graph, median of repetitions",
                "validation": "eager equations with FP64 projections quantized at original dtype boundaries",
                "eager_timing": "original input-dtype projection; batch-dependent rounding recorded separately",
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            }
        )
    )
    for tokens in args.tokens:
        residual, update = rand(tokens, width), rand(tokens, dim)

        def fused():
            mixed, gates = component.read(residual)
            return mixed, component.write(residual, update, gates)

        def reference():
            return eager(component, residual, update)

        actual = fused()
        expected = eager(component, residual, update, precise_projections=True)
        for value, oracle in zip(actual, expected):
            torch.testing.assert_close(value, oracle, atol=2e-3, rtol=torch.finfo(residual.dtype).eps)
        results = {
            "eager_max_abs": [float((value - old).float().abs().max()) for value, old in zip(actual, reference())]
        }
        for name, func in [("eager", reference), ("fused", fused)]:
            samples = [do_bench_cudagraph(func, rep=100) * 1000 for _ in range(args.repetitions)]
            results[name + "_us"] = sorted(samples)[len(samples) // 2]
            results[name + "_samples_us"] = samples
        print(json.dumps({"tokens": tokens, **results, "speedup": results["eager_us"] / results["fused_us"]}))


if __name__ == "__main__":
    main()
