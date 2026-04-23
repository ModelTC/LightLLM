"""Benchmark grouped_matmul_kernel on Qwen3-Omni-30B-A3B (thinker) MoE shapes,
comparing LightLLM vs vLLM vs SGLang Triton fused-MoE implementations.

Matches launch_fp8.sh (--tp 2 --quant_type vllm-fp8w8a8). At this config
(per-channel FP8 w8a8, dynamic per-token activation, no block_shape) on
Blackwell, LightLLM / vLLM / SGLang all dispatch to their Triton fused-MoE path,
so those three columns form an apples-to-apples comparison of grouped-GEMM
kernels. FlashInfer is NOT in the default comparison: on SM120 it has no
per-channel or block-scaled FP8 MoE path, and opting in with
`--providers ...,flashinfer` runs its per-tensor FP8 CUTLASS kernel instead,
shown with a starred column name (`flashinfer*`) to flag the different workload.

Sweeps num_tokens (called batch_size following vLLM's benchmark_moe.py
convention: MoE's GEMM dim M is just num_tokens, the seq dim is irrelevant
once tokens are flattened). Reports median ms via
triton.testing.do_bench_cudagraph.

LightLLM and vLLM share the same (topk_ids, topk_weights) — canonical
softmax + topk (+ renorm), computed once per cell outside the timed region —
so those two columns are strictly apples-to-apples on routing. All
per-provider setup (quant config, LightLLM's in-place scratch buffer,
vLLM's FusedMoEQuantConfig) also happens outside the timed region; only the
fused_experts call itself is timed.

Model / SGLang paths resolve from env vars first, then from the repo-relative
defaults:
    QWEN3_OMNI_MODEL_DIR   fallback: <repo>/../models/Qwen3-Omni-30B-A3B-Instruct
    SGLANG_SRC             fallback: <repo>/../sglang/python

Usage (from LightLLM repo root):
    python test/benchmark/kernel/benchmark_qwen3_omni_grouped_matmul.py
    python test/benchmark/kernel/benchmark_qwen3_omni_grouped_matmul.py \\
        --batch-sizes 64,512,4096
    python test/benchmark/kernel/benchmark_qwen3_omni_grouped_matmul.py \\
        --providers lightllm,vllm                          # skip sglang
    python test/benchmark/kernel/benchmark_qwen3_omni_grouped_matmul.py --quant-mode block128
    python test/benchmark/kernel/benchmark_qwen3_omni_grouped_matmul.py \\
        --providers lightllm,vllm,flashinfer --flashinfer-per-tensor-fallback
    python test/benchmark/kernel/benchmark_qwen3_omni_grouped_matmul.py \\
        --providers lightllm,vllm --batch-sizes 512 --profile
"""
import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import torch
import triton

from lightllm.common.basemodel.triton_kernel.fused_moe.grouped_fused_moe import (
    fused_experts as lightllm_fused_experts,
)


def _repo_relative(*parts: str) -> str:
    # This file lives at <repo>/test/benchmark/kernel/benchmark_*.py, so the
    # LightLLM repo root is three levels up from __file__.
    return str(Path(__file__).resolve().parents[3].joinpath(*parts))


DEFAULT_MODEL_DIR = os.environ.get(
    "QWEN3_OMNI_MODEL_DIR",
    _repo_relative("..", "models", "Qwen3-Omni-30B-A3B-Instruct"),
)
DEFAULT_SGLANG_SRC = os.environ.get(
    "SGLANG_SRC",
    _repo_relative("..", "sglang", "python"),
)

try:
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts as vllm_fused_experts
    from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig

    _HAS_VLLM = True
    _VLLM_IMPORT_ERR = None
except Exception as _e:
    _HAS_VLLM = False
    _VLLM_IMPORT_ERR = repr(_e)

try:
    from flashinfer.fused_moe import cutlass_fused_moe as flashinfer_cutlass_fused_moe

    _HAS_FLASHINFER = True
    _FLASHINFER_IMPORT_ERR = None
except Exception as _e:
    _HAS_FLASHINFER = False
    _FLASHINFER_IMPORT_ERR = repr(_e)

if DEFAULT_SGLANG_SRC and DEFAULT_SGLANG_SRC not in sys.path:
    sys.path.append(DEFAULT_SGLANG_SRC)

# sglang/__init__.py eagerly imports IPython; stub it if unavailable so the
# real blocker (usually sgl_kernel C++ ABI mismatch) surfaces instead.
import types as _types

for _mod in ("IPython", "IPython.display"):
    if _mod not in sys.modules:
        _stub = _types.ModuleType(_mod)
        _stub.HTML = object
        _stub.display = lambda *a, **k: None
        sys.modules[_mod] = _stub

try:
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
        fused_moe as sglang_fused_moe,
    )
    from sglang.srt.layers.moe.topk import select_experts as sglang_select_experts, TopKConfig

    _HAS_SGLANG = True
    _SGLANG_IMPORT_ERR = None
except Exception as _e:
    _HAS_SGLANG = False
    _SGLANG_IMPORT_ERR = repr(_e)


def load_thinker_moe_config(model_dir: str) -> dict:
    with open(Path(model_dir) / "config.json") as f:
        cfg = json.load(f)
    tc = cfg["thinker_config"]["text_config"]
    return {
        "hidden_size": tc["hidden_size"],
        "moe_intermediate_size": tc["moe_intermediate_size"],
        "num_experts": tc["num_experts"],
        "topk": tc["num_experts_per_tok"],
        "num_hidden_layers": tc["num_hidden_layers"],
        "norm_topk_prob": tc.get("norm_topk_prob", True),
    }


def parse_str_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_int_list(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _randn(shape, dtype, device, gen):
    # Use torch.empty + in-place normal_ with an explicit generator so we
    # never touch the default philox state (cudagraph captures corrupt it).
    return torch.empty(shape, dtype=dtype, device=device).normal_(generator=gen)


def _rand(shape, dtype, device, gen):
    return torch.empty(shape, dtype=dtype, device=device).uniform_(generator=gen)


def build_inputs(num_tokens, cfg, tp_size, dtype, quant_mode, device="cuda", gen=None):
    if gen is None:
        gen = torch.Generator(device=device).manual_seed(0)
    E = cfg["num_experts"]
    H = cfg["hidden_size"]
    shard_N = 2 * cfg["moe_intermediate_size"] // tp_size
    inter_per_rank = cfg["moe_intermediate_size"] // tp_size

    x = _randn((num_tokens, H), dtype, device, gen)
    gate = _randn((num_tokens, E), torch.float32, device, gen)

    if quant_mode == "none":
        w1 = _randn((E, shard_N, H), dtype, device, gen)
        w2 = _randn((E, H, inter_per_rank), dtype, device, gen)
        return dict(
            x=x,
            gate=gate,
            w1=w1,
            w2=w2,
            use_fp8=False,
            w1_scale=None,
            w2_scale=None,
            a1_scale=None,
            a2_scale=None,
        )

    w1 = (_randn((E, shard_N, H), dtype, device, gen) * 0.01).to(torch.float8_e4m3fn)
    w2 = (_randn((E, H, inter_per_rank), dtype, device, gen) * 0.01).to(torch.float8_e4m3fn)

    if quant_mode == "per-channel":
        w1_scale = _rand((E, shard_N), torch.float32, device, gen) * 0.01 + 0.01
        w2_scale = _rand((E, H), torch.float32, device, gen) * 0.01 + 0.01
    else:
        bn = bk = 128
        w1_scale = (
            _rand(
                (E, (shard_N + bn - 1) // bn, (H + bk - 1) // bk),
                torch.float32,
                device,
                gen,
            )
            * 0.01
            + 0.01
        )
        w2_scale = (
            _rand(
                (E, (H + bn - 1) // bn, (inter_per_rank + bk - 1) // bk),
                torch.float32,
                device,
                gen,
            )
            * 0.01
            + 0.01
        )

    return dict(
        x=x,
        gate=gate,
        w1=w1,
        w2=w2,
        use_fp8=True,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=None,
        a2_scale=None,
    )


def compute_shared_topk(inputs, cfg):
    """Canonical router output shared across providers.

    softmax + topk (+ renorm). Returns fp32 weights and int32 ids, which are
    the dtypes both LightLLM's and vLLM's fused_experts expect, so the same
    tensors can be fed to each without per-provider conversion.
    """
    scores = torch.softmax(inputs["gate"], dim=-1)
    topk_w, topk_ids = torch.topk(scores, k=cfg["topk"], dim=-1)
    if cfg["norm_topk_prob"]:
        topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)
    return {
        "topk_weights": topk_w.to(torch.float32).contiguous(),
        "topk_ids": topk_ids.to(torch.int32).contiguous(),
    }


def prepare_lightllm(inputs, cfg, quant_mode, shared):
    # LightLLM's fused_experts(inplace=True) writes through hidden_states;
    # its inplace=False path hits a torch.ops schema bug in this build
    # (outplace_fused_experts_impl declares return=None but the impl returns
    # a Tensor). Pre-allocate a scratch buffer here so the timed loop doesn't
    # pay ~0.5 ms of D2D clone per iteration. The buffer is mutated across
    # graph replays; that is safe because MoE grouped_matmul timing is
    # value-agnostic (control flow depends on topk_ids, not on activations).
    return {
        "topk_weights": shared["topk_weights"],
        "topk_ids": shared["topk_ids"],
        "x_buf": inputs["x"].clone(),
    }


def run_lightllm(inputs, cfg, quant_mode, prepared):
    return lightllm_fused_experts(
        hidden_states=prepared["x_buf"],
        w1=inputs["w1"],
        w2=inputs["w2"],
        topk_weights=prepared["topk_weights"],
        topk_ids=prepared["topk_ids"],
        inplace=True,
        use_fp8_w8a8=inputs["use_fp8"],
        w1_scale=inputs["w1_scale"],
        w2_scale=inputs["w2_scale"],
    )


def prepare_vllm(inputs, cfg, quant_mode, shared):
    if inputs["use_fp8"]:
        block_shape = [128, 128] if quant_mode == "block128" else None
        qcfg = FusedMoEQuantConfig.make(
            quant_dtype=torch.float8_e4m3fn,
            per_act_token_quant=(block_shape is None),
            per_out_ch_quant=(block_shape is None),
            block_shape=block_shape,
            w1_scale=inputs["w1_scale"],
            w2_scale=inputs["w2_scale"],
            a1_scale=None,
            a2_scale=None,
        )
    else:
        qcfg = None

    return {
        "topk_weights": shared["topk_weights"],
        "topk_ids": shared["topk_ids"],
        "qcfg": qcfg,
    }


def run_vllm(inputs, cfg, quant_mode, prepared):
    return vllm_fused_experts(
        hidden_states=inputs["x"],
        w1=inputs["w1"],
        w2=inputs["w2"],
        topk_weights=prepared["topk_weights"],
        topk_ids=prepared["topk_ids"],
        inplace=False,
        global_num_experts=inputs["w1"].shape[0],
        quant_config=prepared["qcfg"],
    )


def prepare_sglang(inputs, cfg, quant_mode, shared):
    # sglang is currently unavailable in this workspace (sgl_kernel C++ ABI
    # mismatch), so this path is not exercised. It still computes its own
    # routing via sglang_select_experts rather than the shared topk; plumbing
    # shared into a StandardTopKOutput here would be easy, but without a live
    # sglang to validate against we leave the original behaviour.
    topk_out = sglang_select_experts(
        hidden_states=inputs["x"],
        router_logits=inputs["gate"],
        topk_config=TopKConfig(top_k=cfg["topk"], renormalize=cfg["norm_topk_prob"]),
    )
    return {"topk_output": topk_out}


def run_sglang(inputs, cfg, quant_mode, prepared):
    block_shape = [128, 128] if quant_mode == "block128" else None
    return sglang_fused_moe(
        hidden_states=inputs["x"],
        w1=inputs["w1"],
        w2=inputs["w2"],
        topk_output=prepared["topk_output"],
        use_fp8_w8a8=inputs["use_fp8"],
        per_channel_quant=(inputs["use_fp8"] and block_shape is None),
        w1_scale=inputs["w1_scale"],
        w2_scale=inputs["w2_scale"],
        a1_scale=None,
        a2_scale=None,
        block_shape=block_shape,
    )


def prepare_flashinfer(inputs, cfg, quant_mode, shared):
    # flashinfer is currently unavailable on this hardware/SM combo; this
    # path is not exercised. The shared (fp32, int32) tensors are already in
    # the format flashinfer wants, so we just forward them.
    return {
        "topk_weights": shared["topk_weights"],
        "topk_ids": shared["topk_ids"],
    }


def run_flashinfer(inputs, cfg, quant_mode, prepared, allow_per_tensor_fp8_fallback=False):
    # FlashInfer on Blackwell SM120 (RTX 5090):
    #   - trtllm_fp8_block_scale_moe: no SM120 cubins
    #   - cutlass_fused_moe with use_deepseek_fp8_block_scale=True: NYI on Blackwell
    #   - cutlass_fused_moe with per-tensor FP8: works (JITs a SM120 .so)
    #   - cutlass_fused_moe with bf16: works
    # So bf16 ("none") runs as-is. Per-channel / block128 have no native
    # flashinfer match on SM120; we refuse to pretend they do, and only fall
    # back to per-tensor FP8 when --flashinfer-per-tensor-fallback is set.
    # In that case the column header is 'flashinfer*' to flag the different
    # workload vs the LightLLM/vLLM/SGLang per-channel / block128 numbers.
    topk_w = prepared["topk_weights"]
    topk_ids = prepared["topk_ids"]

    if not inputs["use_fp8"]:
        out = flashinfer_cutlass_fused_moe(
            input=inputs["x"],
            token_selected_experts=topk_ids,
            token_final_scales=topk_w,
            fc1_expert_weights=inputs["w1"],
            fc2_expert_weights=inputs["w2"],
            output_dtype=inputs["x"].dtype,
            quant_scales=[],
        )
        return out[0] if isinstance(out, list) else out

    if not allow_per_tensor_fp8_fallback:
        raise RuntimeError(
            f"flashinfer has no SM120 MoE path for --quant-mode {quant_mode!r} "
            "(trtllm block-scale lacks SM120 cubins; cutlass block-scale is "
            "NYI on Blackwell). Pass --flashinfer-per-tensor-fallback to run "
            "its per-tensor FP8 CUTLASS kernel instead; that number will NOT "
            "be directly comparable to the other providers' per-channel / "
            "block128 times (the column will be tagged 'flashinfer*')."
        )

    E = inputs["w1"].shape[0]
    device = inputs["x"].device
    x_fp8 = inputs["x"].to(torch.float8_e4m3fn)
    g1 = torch.full((E,), 1e-4, dtype=torch.float32, device=device)
    g2 = torch.full((E,), 1e-4, dtype=torch.float32, device=device)
    a1_sc = torch.tensor(0.01, dtype=torch.float32, device=device)
    a2_gs = torch.tensor(100.0, dtype=torch.float32, device=device)
    out = flashinfer_cutlass_fused_moe(
        input=x_fp8,
        token_selected_experts=topk_ids,
        token_final_scales=topk_w,
        fc1_expert_weights=inputs["w1"],
        fc2_expert_weights=inputs["w2"],
        output_dtype=inputs["x"].dtype,
        quant_scales=[g1, a2_gs, g2, a1_sc],
    )
    return out[0] if isinstance(out, list) else out


PROVIDERS = {
    "lightllm": run_lightllm,
    "vllm": run_vllm,
    "sglang": run_sglang,
    "flashinfer": run_flashinfer,
}
PROVIDER_PREPARE = {
    "lightllm": prepare_lightllm,
    "vllm": prepare_vllm,
    "sglang": prepare_sglang,
    "flashinfer": prepare_flashinfer,
}
PROVIDER_AVAIL = {
    "lightllm": True,
    "vllm": _HAS_VLLM,
    "sglang": _HAS_SGLANG,
    "flashinfer": _HAS_FLASHINFER,
}
PROVIDER_ERR = {
    "lightllm": None,
    "vllm": _VLLM_IMPORT_ERR,
    "sglang": _SGLANG_IMPORT_ERR,
    "flashinfer": _FLASHINFER_IMPORT_ERR,
}


def profile_cudagraph_replay(fn, label):
    # Profile graph replay only so the table reflects cudagraph execution.
    # Capture a single provider call; the benchmark path below still uses
    # triton.testing.do_bench_cudagraph's repeated capture for stable timing.
    print(f"\n[profile] {label}")

    with torch.cuda.stream(torch.cuda.Stream()):
        fn()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        torch.cuda.synchronize()

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
        ) as prof:
            graph.replay()
            torch.cuda.synchronize()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))


def bench_provider(
    fn,
    inputs,
    cfg,
    quant_mode,
    prepared,
    warmup,
    fn_kwargs=None,
    profile=False,
    profile_label=None,
):
    fn_kwargs = fn_kwargs or {}
    for _ in range(warmup):
        fn(inputs, cfg, quant_mode, prepared, **fn_kwargs)
    torch.cuda.synchronize()

    def _closure():
        return fn(inputs, cfg, quant_mode, prepared, **fn_kwargs)

    if profile:
        profile_cudagraph_replay(_closure, profile_label or fn.__name__)

    return triton.testing.do_bench_cudagraph(_closure, rep=50)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    # For MoE the only workload axis is num_tokens (the GEMM's M); we follow
    # vLLM's benchmark_moe.py and call it --batch-sizes even though it's
    # really "tokens per forward". Default list matches vLLM's tuner sweep.
    ap.add_argument(
        "--batch-sizes",
        type=parse_int_list,
        default=[
            1,
            2,
            4,
            8,
            16,
            24,
            32,
            48,
            64,
            96,
            128,
            256,
            512,
            1024,
            1536,
            2048,
            3072,
            4096,
        ],
    )
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument(
        "--quant-mode",
        choices=["none", "per-channel", "block128"],
        default="per-channel",
        help="per-channel matches --quant_type vllm-fp8w8a8 in launch_fp8.sh",
    )
    ap.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument(
        "--profile",
        action="store_true",
        default=False,
        help=(
            "Profile one cudagraph replay after warmup and print "
            "prof.key_averages().table(sort_by='cuda_time_total', row_limit=20). "
            "This runs once per provider per batch size."
        ),
    )
    ap.add_argument(
        "--providers",
        type=parse_str_list,
        default=["lightllm", "vllm", "sglang"],
        help=(
            "comma-separated subset of lightllm,vllm,sglang,flashinfer. "
            "flashinfer is NOT in the default list because it has no per-channel "
            "or block-scaled FP8 MoE path on SM120; opt in explicitly and (for "
            "FP8 modes) pair with --flashinfer-per-tensor-fallback."
        ),
    )
    ap.add_argument(
        "--flashinfer-per-tensor-fallback",
        action="store_true",
        default=False,
        help=(
            "When flashinfer is requested for an FP8 quant-mode that has no "
            "SM120 match (per-channel, block128), fall back to its per-tensor "
            "FP8 CUTLASS kernel. The column is tagged 'flashinfer*' to flag "
            "that the workload differs from the other providers."
        ),
    )
    args = ap.parse_args()

    torch.cuda.manual_seed_all(0)

    cfg = load_thinker_moe_config(args.model_dir)
    cfg["inter_per_rank"] = cfg["moe_intermediate_size"] // args.tp
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    providers = []
    for p in args.providers:
        if p not in PROVIDERS:
            print(f"[skip] unknown provider '{p}' (choices: lightllm, vllm, sglang, flashinfer)")
        elif not PROVIDER_AVAIL[p]:
            print(f"[skip] {p} unavailable: {PROVIDER_ERR[p]}")
        else:
            providers.append(p)
    if not providers:
        print("no providers available; aborting")
        return

    print(f"[config] thinker MoE: {cfg}")
    print(f"[config] tp={args.tp} quant={args.quant_mode} dtype={args.dtype}")
    print(f"[config] sweep: batch_sizes (num_tokens) = {args.batch_sizes}")
    print(f"[config] providers: {providers}\n")
    if args.profile:
        print("[config] torch profiler enabled (one cudagraph replay per provider per batch size)\n")

    # Tag the flashinfer column so readers notice when the per-tensor FP8
    # fallback is engaged (it's a different workload from per-channel/block128).
    flashinfer_starred = (
        "flashinfer" in providers
        and args.flashinfer_per_tensor_fallback
        and args.quant_mode in ("per-channel", "block128")
    )
    column_labels = {p: (p + "*" if (p == "flashinfer" and flashinfer_starred) else p) for p in providers}
    prov_col = "".join(f"{column_labels[p] + '(ms)':>15}" for p in providers)
    header = f"{'tokens':>8} |{prov_col}"
    print(header)
    print("-" * len(header))
    if flashinfer_starred:
        print("  * flashinfer runs per-tensor FP8 CUTLASS (not per-channel / " "block128); not directly comparable.")

    results: Dict = {}
    for num_tokens in args.batch_sizes:
        # Reset philox state left dirty by triton.testing.do_bench_cudagraph.
        # (Note: build_inputs / compute_shared_topk use their own explicit
        # Generator, so this seed reset is cosmetic -- kept only to be
        # defensive against future callers that touch the default state.)
        torch.cuda.synchronize()
        torch.cuda.manual_seed_all(0)
        inputs = build_inputs(num_tokens, cfg, args.tp, dtype, args.quant_mode)
        shared = compute_shared_topk(inputs, cfg)

        # Precompute each provider's per-provider setup (quant config,
        # LightLLM's in-place scratch buffer, etc.) outside the timed region
        # so the reported ms reflects the fused_experts kernel cost only.
        prepared: Dict = {}
        for p in providers:
            try:
                prepared[p] = PROVIDER_PREPARE[p](inputs, cfg, args.quant_mode, shared)
            except Exception as e:
                prepared[p] = None
                print(f"\n[prepare-error] provider={p} tokens={num_tokens}: {type(e).__name__}: {e}")

        row_cells = []
        for p in providers:
            if prepared[p] is None:
                row_cells.append(f"{'ERR':>15}")
                continue
            fn_kwargs = (
                {"allow_per_tensor_fp8_fallback": args.flashinfer_per_tensor_fallback} if p == "flashinfer" else {}
            )
            try:
                ms = bench_provider(
                    PROVIDERS[p],
                    inputs,
                    cfg,
                    args.quant_mode,
                    prepared[p],
                    args.warmup,
                    fn_kwargs=fn_kwargs,
                    profile=args.profile,
                    profile_label=(f"provider={p} tokens={num_tokens} quant={args.quant_mode} " f"dtype={args.dtype}"),
                )
                row_cells.append(f"{ms:>15.3f}")
                results[(p, num_tokens)] = ms
            except Exception as e:
                row_cells.append(f"{'ERR':>15}")
                print(f"\n[error] provider={p} tokens={num_tokens}: {type(e).__name__}: {e}")
            gc.collect()
            torch.cuda.empty_cache()

        print(f"{num_tokens:>8} |{''.join(row_cells)}")

        del inputs, prepared, shared
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
