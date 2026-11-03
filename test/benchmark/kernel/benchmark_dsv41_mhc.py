"""Compare eager and deferred FFN post at the actual V4.1 cross-layer boundary."""

import argparse
import json
import statistics
from types import SimpleNamespace

import torch
from triton.testing import do_bench_cudagraph

from lightllm.models.deepseek_v41.layer_infer.transformer_layer_infer import DeepseekV41TransformerLayerInfer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 6, 16, 32, 64, 128, 384, 640, 768])
    args = parser.parse_args()
    torch.manual_seed(410)
    hidden = 5120
    for tokens in args.rows:
        residual = torch.randn(tokens, 4 * hidden, dtype=torch.bfloat16, device="cuda")
        y = torch.randn(tokens, hidden, dtype=torch.bfloat16, device="cuda")
        mix = torch.randn(tokens, 4, 4, device="cuda").softmax(-1)
        post, previous = torch.rand(tokens, 4, device="cuda"), torch.rand(tokens, 4, device="cuda")
        fn = torch.randn(24, hidden * 4, device="cuda") * 0.005
        scale, base = torch.randn(3, device="cuda"), torch.randn(24, device="cuda")
        norm = torch.randn(hidden, dtype=torch.bfloat16, device="cuda")
        layer = DeepseekV41TransformerLayerInfer.__new__(DeepseekV41TransformerLayerInfer)
        layer.hc_mult = 4
        layer.eps_ = layer.hc_eps = 1e-6
        layer.sinkhorn_iters = 20
        layer.engram_index = None
        layer.is_last_layer = False
        weight = SimpleNamespace(
            hc_attn_fn_=SimpleNamespace(weight=fn),
            hc_attn_scale_=SimpleNamespace(weight=scale),
            hc_attn_base_=SimpleNamespace(weight=base),
            attn_norm_=SimpleNamespace(weight=norm),
        )

        def separate():
            layer.defer_hc_post = False
            completed = layer._hc_ffn_out(y, residual, mix, post, previous)
            return layer._hc_attn_in(completed, None, weight)

        def fused():
            layer.defer_hc_post = True
            pending = layer._hc_ffn_out(y, residual, mix, post, previous)
            return layer._hc_attn_in(pending, None, weight)

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
