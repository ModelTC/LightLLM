import argparse
import json
import math
import os

import torch
import torch.nn.functional as F
import triton
from triton.testing import do_bench_cudagraph

from lightllm.common.ngram_embedding.ple import PLELayer


def eager(layer, residual, embeddings, history):
    batch, length, width = residual.shape

    def norm(x, weight):
        grouped = x.float().unflatten(-1, (layer.branches, -1))
        normalized = grouped * torch.rsqrt(grouped.square().mean(-1, keepdim=True) + layer.eps)
        return (normalized.flatten(-2) * (1 + weight.float())).to(x.dtype)

    key = norm(F.linear(embeddings, layer.key_weight), layer.norm_key).unflatten(-1, (layer.branches, -1))
    query = norm(residual, layer.norm_query).unflatten(-1, (layer.branches, -1))
    value = F.linear(embeddings, layer.value_weight)
    gate = (key * query).sum(-1, keepdim=True) / math.sqrt(layer.hidden_size)
    gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
    gated = (torch.sigmoid(gate) * value.unsqueeze(-2)).flatten(-2)
    conv_inputs = norm(gated, layer.norm_conv)
    full = torch.cat((history, conv_inputs), dim=1)
    conv = F.conv1d(full.transpose(1, 2), layer.conv_weight.unsqueeze(1), dilation=layer.dilation, groups=width)
    output = gated + F.silu(conv).transpose(1, 2)
    return output, full[:, -layer.context_len :].contiguous()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", nargs="+", default=["1x1", "16x1", "1x128", "4x128"])
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if not os.environ.get("EXP_ACTIVE"):
        parser.error('Wrap this benchmark with exp -m "purpose"')
    torch.manual_seed(137)
    dim, branches, embed_dim = 2560, 4, 2560
    width = dim * branches

    def rand(*shape):
        return torch.randn(*shape, device="cuda", dtype=torch.bfloat16)

    layer = PLELayer(
        rand(width, embed_dim) / embed_dim**0.5,
        rand(dim, embed_dim) / embed_dim**0.5,
        rand(width) * 0.1,
        rand(width) * 0.1,
        rand(width) * 0.1,
        rand(width, 4) * 0.1,
        branches,
        3,
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
                "embedding_width": embed_dim,
                "kernel_size": 4,
                "dilation": 3,
                "timing": "CUDA graph, median of interleaved repetitions",
                "includes_table_lookup": False,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            }
        ),
        flush=True,
    )
    for shape in args.shapes:
        batch, length = map(int, shape.split("x"))
        residual = rand(batch, length, width)
        embeddings = rand(batch, length, embed_dim)
        history = layer.empty_history(batch).normal_()
        starts = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * length

        def fused():
            output, conv_inputs = layer.forward(
                residual.view(-1, width), embeddings.view(-1, embed_dim), starts, history
            )
            return output.view(batch, length, width), layer.history_after(conv_inputs, starts, history)

        def reference():
            return eager(layer, residual, embeddings, history)

        for actual, expected in zip(fused(), reference()):
            torch.testing.assert_close(
                actual, expected, atol=2 * torch.finfo(residual.dtype).eps, rtol=2 * torch.finfo(residual.dtype).eps
            )
        samples = {"eager": [], "fused": []}
        for repeat in range(args.repetitions):
            functions = [("eager", reference), ("fused", fused)]
            for name, func in functions[:: 1 if repeat % 2 == 0 else -1]:
                samples[name].append(do_bench_cudagraph(func, rep=100) * 1000)
        medians = {name + "_us": sorted(values)[len(values) // 2] for name, values in samples.items()}
        print(
            json.dumps(
                {
                    "batch": batch,
                    "tokens_per_request": length,
                    **medians,
                    "speedup": medians["eager_us"] / medians["fused_us"],
                    "samples_us": samples,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
