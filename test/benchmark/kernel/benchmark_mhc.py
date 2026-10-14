"""Compare shared mHC with the DeepSeek-V4 vLLM/TileLang path.

Run from the repository root in a CUDA environment with vLLM installed:
  PYTHONPATH=. python test/benchmark/kernel/benchmark_mhc.py --output /tmp/mhc.json

To also compare the original shared implementation, snapshot its pre_norm.py and
post.py as pre_norm_baseline.py and post_baseline.py outside the repository before
editing, then pass --baseline-dir /path/to/snapshot. No baseline kernels are kept
in the repository. Timings include the entire pre + RMSNorm or post -> pre +
RMSNorm path, using CUDA graphs after compilation and correctness checks.

Use LIGHTLLM_TRITON_AUTOTUNE_LEVEL=1 and --autotune to search configurations
before measuring. Tuning/compilation time is excluded from the reported latency.
"""

import argparse
import importlib.metadata
import importlib.util
import json
from pathlib import Path
from statistics import median
import sys
from types import ModuleType

import torch
import triton
import triton.testing

from lightllm.models.deepseek_v4.layer_infer import hyper_connection as shared
from lightllm.common.triton_utils.autotuner import Autotuner, AutotuneLevel
from lightllm.common.basemodel.triton_kernel.mhc import head, post, pre_norm
from lightllm.utils.envs_utils import get_triton_autotune_level


def load_baseline(root, name):
    package_name = "baseline_mhc"
    if package_name not in sys.modules:
        package = ModuleType(package_name)
        package.__path__ = [str(root)]
        sys.modules[package_name] = package
    module_name = package_name + "." + name
    if module_name in sys.modules:
        return sys.modules[module_name]
    if name == "pre_norm":
        load_baseline(root, "post")
    spec = importlib.util.spec_from_file_location(module_name, root / (name + "_baseline.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", nargs="+", type=int, default=[1, 8, 16, 32, 128, 512, 2048, 8192])
    parser.add_argument("--hidden-sizes", nargs="+", type=int, default=[4096])
    parser.add_argument("--rep", type=int, default=80, help="CUDA graph timing duration in milliseconds")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--autotune", action="store_true", help="Tune shared mHC in the normal LightLLM warmup phase")
    args = parser.parse_args()
    if min(args.tokens + args.hidden_sizes + [args.rep, args.rounds]) < 1:
        parser.error("tokens, hidden sizes, rep and rounds must be positive")
    if args.autotune and get_triton_autotune_level() not in [
        AutotuneLevel.ADAPTIVE_AUTOTUNE,
        AutotuneLevel.FORCE_AUTOTUNE,
    ]:
        parser.error("--autotune requires LIGHTLLM_TRITON_AUTOTUNE_LEVEL=1 or 2")

    # Register only the comparison backend; production shared mHC has no vLLM dependency.
    import vllm.model_executor.layers.mhc  # noqa: F401

    original_pre = original_post = original_post_pre = original_head = None
    if args.baseline_dir is not None:
        original_module = load_baseline(args.baseline_dir, "pre_norm")
        original_pre = original_module.hc_pre_norm
        original_post_pre = getattr(original_module, "hc_post_pre_norm", None)
        original_post = load_baseline(args.baseline_dir, "post").hc_post
        if (args.baseline_dir / "head_baseline.py").exists():
            original_head = load_baseline(args.baseline_dir, "head").hc_head

    torch.manual_seed(42)
    triton.set_allocator(lambda size, alignment, stream: torch.empty(size, device="cuda", dtype=torch.int8))
    records = []
    result = {
        "gpu": torch.cuda.get_device_name(),
        "cuda": torch.version.cuda,
        "versions": {name: version(name) for name in ["torch", "triton", "vllm", "tilelang", "deep-gemm"]},
        "timing": "GPU microseconds; median across CUDA-graph median samples; compilation excluded",
        "rep_ms": args.rep,
        "rounds": args.rounds,
        "streams": 4,
        "sinkhorn_iters": 20,
        "rms_eps": 1e-6,
        "hc_eps": 1e-6,
        "dtype": "bfloat16",
        "baseline_dir": str(args.baseline_dir) if args.baseline_dir is not None else None,
        "autotune": args.autotune,
        "records": records,
    }
    print(json.dumps({key: value for key, value in result.items() if key != "records"}), flush=True)

    def measure(op, tokens, hidden, variants):
        values = {name: fn() for name, fn in variants.items()}
        reference = values["shared"]
        for name, tensors in values.items():
            if name == "shared":
                continue
            for index, (actual, expected) in enumerate(zip(tensors, reference)):
                mixing_weight = op in ["pre_norm", "post_pre_norm"] and index >= 2
                torch.testing.assert_close(
                    actual,
                    expected,
                    atol=1e-3 if mixing_weight else 0.04,
                    rtol=2e-3 if mixing_weight else 0.03,
                )
        del values
        torch.cuda.synchronize()
        samples = {name: [] for name in variants}
        for round_index in range(args.rounds):
            names = list(variants)
            if round_index % 2:
                names.reverse()
            for name in names:
                samples[name].append(
                    triton.testing.do_bench_cudagraph(variants[name], rep=args.rep, return_mode="median") * 1000
                )
        timings = {name: median(us) for name, us in samples.items()}
        record = {
            "op": op,
            "tokens": tokens,
            "hidden": hidden,
            "us": timings,
            "samples_us": samples,
            "shared_speedup": {name: us / timings["shared"] for name, us in timings.items() if name != "shared"},
        }
        records.append(record)
        print(json.dumps(record), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    for hidden in args.hidden_sizes:
        for tokens in args.tokens:
            streams, eps = 4, 1e-6
            residual = torch.randn(tokens, streams, hidden, device="cuda", dtype=torch.bfloat16)
            flat = residual.flatten(1)
            fn = torch.randn(24, streams * hidden, device="cuda") * 0.005
            scale = torch.randn(3, device="cuda")
            base = torch.randn(24, device="cuda")
            norm = torch.randn(hidden, device="cuda", dtype=torch.bfloat16)
            layer_output = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)

            def vllm_pre():
                post, mix, x = torch.ops.vllm.mhc_pre_tilelang(
                    residual, fn, scale, base, eps, eps, eps, 2.0, 20, norm_weight=norm, norm_eps=eps
                )
                return x, residual, post, mix

            def shared_pre():
                return shared.hc_pre(residual, fn, scale, base, eps, eps, 20, norm, eps)

            def baseline_pre():
                x, mix, post = original_pre(flat, fn, scale, base, norm, streams, eps, eps, eps, 20)
                return x, residual, post.unsqueeze(-1), mix

            if args.autotune:
                with Autotuner.autotune_warmup():
                    shared_pre()

            variants = {"vllm": vllm_pre, "shared": shared_pre}
            if original_pre is not None:
                variants["original_common"] = baseline_pre
            measure("pre_norm", tokens, hidden, variants)
            _, _, post_mix, residual_mix = shared_pre()

            def vllm_post():
                return (torch.ops.vllm.mhc_post_tilelang(layer_output, residual, post_mix, residual_mix),)

            def shared_post():
                return (shared.hc_post(layer_output, residual, post_mix, residual_mix),)

            def baseline_post():
                x = original_post(layer_output, flat, residual_mix, post_mix.view(tokens, streams), streams)
                return (x.view_as(residual),)

            variants = {"vllm": vllm_post, "shared": shared_post}
            if args.autotune:
                with Autotuner.autotune_warmup():
                    shared_post()
            if original_post is not None:
                variants["original_common"] = baseline_post
            measure("post", tokens, hidden, variants)

            def vllm_post_pre():
                updated, post, mix, x = torch.ops.vllm.mhc_fused_post_pre_tilelang(
                    layer_output,
                    residual,
                    post_mix,
                    residual_mix,
                    fn,
                    scale,
                    base,
                    eps,
                    eps,
                    eps,
                    2.0,
                    20,
                    norm_weight=norm,
                    norm_eps=eps,
                )
                return x, updated, post, mix

            def shared_post_pre():
                return shared.hc_fused_post_pre(
                    layer_output, residual, post_mix, residual_mix, fn, scale, base, eps, eps, 20, norm, eps
                )

            def baseline_post_pre():
                if original_post_pre is not None:
                    x, updated, mix, post = original_post_pre(
                        layer_output,
                        flat,
                        residual_mix,
                        post_mix.view(tokens, streams),
                        fn,
                        scale,
                        base,
                        norm,
                        streams,
                        eps,
                        eps,
                        eps,
                        20,
                    )
                    return x, updated.view_as(residual), post.unsqueeze(-1), mix
                updated = baseline_post()[0]
                x, mix, post = original_pre(updated.flatten(1), fn, scale, base, norm, streams, eps, eps, eps, 20)
                return x, updated, post.unsqueeze(-1), mix

            variants = {"vllm": vllm_post_pre, "shared": shared_post_pre}
            if args.autotune:
                with Autotuner.autotune_warmup():
                    shared_post_pre()
            if original_pre is not None:
                variants["original_common"] = baseline_post_pre
            measure("post_pre_norm", tokens, hidden, variants)

            head_fn, head_scale, head_base = (
                fn[:streams].contiguous(),
                scale[:1].contiguous(),
                base[:streams].contiguous(),
            )

            def vllm_head():
                out = torch.empty(tokens, hidden, device="cuda", dtype=torch.bfloat16)
                torch.ops.vllm.hc_head_fused_kernel_tilelang(
                    residual, head_fn, head_scale, head_base, out, hidden, eps, eps, streams
                )
                return (out,)

            def shared_head():
                return (shared.hc_head(flat, head_fn, head_scale, head_base, streams, hidden, eps, eps, torch.empty),)

            def baseline_head():
                return (original_head(flat, head_fn, head_scale, head_base, streams, eps, eps),)

            if args.autotune:
                with Autotuner.autotune_warmup():
                    shared_head()
            variants = {"vllm": vllm_head, "shared": shared_head}
            if original_head is not None:
                variants["original_common"] = baseline_head
            measure("head", tokens, hidden, variants)

    if args.autotune:
        tuners = [
            pre_norm._hc_prenorm_projection,
            pre_norm._finish_pre_norm,
            pre_norm._hc_post_pre_norm_decode,
            post.hc_post,
            head.hc_head,
        ]
        result["tuned_configs"] = {
            tuner.kernel_name: [
                {"static_key": dict(key), "by_tokens": configs} for key, configs in tuner.cached_configs.items()
            ]
            for tuner in tuners
        }
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
