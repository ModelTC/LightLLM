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
    python test/benchmark/kernel/benchmark_qwen3_omni_grouped_matmul.py \\
        --tune --tune-level 2 --batch-sizes 512,4096
"""
import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch.distributed as dist
import triton

from lightllm.common.basemodel.triton_kernel.fused_moe.grouped_fused_moe import (
    fused_experts as lightllm_fused_experts,
)
from lightllm.common.triton_utils.autotuner import Autotuner
from lightllm.utils.envs_utils import get_triton_autotune_level


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

    try:
        from flashinfer.fused_moe import trtllm_fp8_block_scale_moe as flashinfer_trtllm_fp8_block_scale_moe
    except Exception:
        flashinfer_trtllm_fp8_block_scale_moe = None

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


def parse_quant_mode(s: str) -> str:
    aliases = {
        "none": "none",
        "per-channel": "per-channel",
        "block128": "block128",
        "vllm-fp8w8a8": "per-channel",
        "vllm-fp8w8a8-b128": "block128",
    }
    if s not in aliases:
        raise argparse.ArgumentTypeError(
            "expected one of: none, per-channel, block128, vllm-fp8w8a8, vllm-fp8w8a8-b128"
        )
    return aliases[s]


def tp_size_per_dp(args) -> int:
    if args.tp % args.dp != 0:
        raise ValueError(f"--tp must be divisible by --dp, got tp={args.tp}, dp={args.dp}")
    return args.tp // args.dp


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
    out = {
        "topk_weights": shared["topk_weights"],
        "topk_ids": shared["topk_ids"],
    }
    if quant_mode == "block128" and inputs["use_fp8"]:
        # trtllm_fp8_block_scale_moe needs the activation pre-quantized to fp8
        # with per-token / per-128-K-block scales in [H//128, M] (column-major)
        # layout. The values we feed are random — benchmark cares about timing,
        # not numerics, just like the random fp8 weights / scales above.
        x = inputs["x"]
        M, H = x.shape
        out["x_fp8"] = (x.to(torch.float32) * 0.1).to(torch.float8_e4m3fn).contiguous()
        out["x_scale"] = torch.full((H // 128, M), 0.1, dtype=torch.float32, device=x.device)
    return out


def run_flashinfer(inputs, cfg, quant_mode, prepared, allow_per_tensor_fp8_fallback=False):
    # FlashInfer paths on Blackwell SM120 (RTX 5090):
    #   - cutlass_fused_moe(bf16): native SM120 module, runs as-is.
    #   - cutlass_fused_moe(use_deepseek_fp8_block_scale=True): user-side raise
    #       'FP8 block scaling not yet implemented for Blackwell.' (the SM120
    #       cutlass module is not compiled with -DENABLE_FP8_BLOCK_SCALE).
    #   - trtllm_fp8_block_scale_moe: the JIT module is built with
    #       supported_major_versions=[10] (SM100/103 only); on SM120 it relies
    #       on PTX -> sm_120 driver-side JIT. Performance / correctness aren't
    #       validated by FlashInfer CI, so we tag the column 'flashinfer^'.
    #   - cutlass_fused_moe(per-tensor FP8): works (SM120 module supports it).
    # Path selection here (must match the column-label logic in main()):
    #   none      -> cutlass_fused_moe(bf16)             column 'flashinfer'
    #   block128  -> trtllm_fp8_block_scale_moe          column 'flashinfer^'
    #     (if --flashinfer-per-tensor-fallback set, override with per-tensor)
    #   per-channel + --flashinfer-per-tensor-fallback   column 'flashinfer*'
    #   per-channel without that flag                    raise
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

    if quant_mode == "block128" and not allow_per_tensor_fp8_fallback:
        if flashinfer_trtllm_fp8_block_scale_moe is None:
            raise RuntimeError(
                "flashinfer.fused_moe.trtllm_fp8_block_scale_moe not importable; "
                "pass --flashinfer-per-tensor-fallback to drop to per-tensor FP8."
            )
        E = inputs["w1"].shape[0]
        device = inputs["x"].device
        rb = torch.zeros(E, dtype=torch.float32, device=device)
        try:
            out = flashinfer_trtllm_fp8_block_scale_moe(
                routing_logits=inputs["gate"],
                routing_bias=rb,
                hidden_states=prepared["x_fp8"],
                hidden_states_scale=prepared["x_scale"],
                gemm1_weights=inputs["w1"],
                gemm1_weights_scale=inputs["w1_scale"],
                gemm2_weights=inputs["w2"],
                gemm2_weights_scale=inputs["w2_scale"],
                num_experts=E,
                top_k=cfg["topk"],
                n_group=None,
                topk_group=None,
                intermediate_size=cfg["inter_per_rank"],
                local_expert_offset=0,
                local_num_experts=E,
                routed_scaling_factor=None,
            )
            return out[0] if isinstance(out, list) else out
        except RuntimeError as e:
            # The trtllm gen module is JIT-built with supported_major_versions=[10]
            # only (datacenter Blackwell B200/GB200). On consumer Blackwell (SM120,
            # RTX 5090) FlashInfer's JitSpec explicitly refuses to build it:
            #   "No supported CUDA architectures found for major versions [10]."
            # No PTX -> sm_120 fallback exists. Surface a clear message rather
            # than letting the raw RuntimeError leak through.
            msg = str(e)
            if "supported CUDA architectures" in msg or "major versions [10]" in msg:
                raise RuntimeError(
                    "flashinfer.trtllm_fp8_block_scale_moe is JIT-built with "
                    "supported_major_versions=[10] only (B200/GB200). On this "
                    "device (cap.major != 10) the build is rejected up-front; "
                    "no PTX -> sm_120 path is generated. There is no native "
                    "flashinfer block-scale MoE for this GPU. Pass "
                    "--flashinfer-per-tensor-fallback to fall back to per-tensor "
                    "FP8 CUTLASS (column will be tagged 'flashinfer*')."
                ) from e
            raise

    if not allow_per_tensor_fp8_fallback:
        raise RuntimeError(
            f"flashinfer has no SM120 MoE path for --quant-mode {quant_mode!r} "
            "(cutlass block-scale is NYI on Blackwell). Pass "
            "--flashinfer-per-tensor-fallback to run its per-tensor FP8 CUTLASS "
            "kernel instead; that number will NOT be directly comparable to the "
            "other providers' per-channel times (the column will be tagged "
            "'flashinfer*')."
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

    try:
        return triton.testing.do_bench_cudagraph(_closure, rep=50), False
    except Exception as e:
        # Some kernels (e.g. flashinfer's per-tensor cutlass MoE on first call,
        # whose internal autotuner allocates / probes GPU state) break the
        # cudagraph capture (cudaErrorStreamCaptureInvalidated). Fall back to
        # plain do_bench (event-based timing) so the benchmark still produces a
        # number, with the caller free to mark the column with a tilde.
        msg = str(e)
        if "cudaErrorStreamCaptureInvalidated" not in msg and "during capture" not in msg:
            raise
        torch.cuda.synchronize()
        return triton.testing.do_bench(_closure, rep=50), True


def _is_torchrun():
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def _rank0():
    return not dist.is_initialized() or dist.get_rank() == 0


def _log_rank0(message):
    if _rank0():
        print(message)


def _set_triton_autotune_level(value):
    if value is None:
        os.environ.pop("LIGHTLLM_TRITON_AUTOTUNE_LEVEL", None)
    else:
        os.environ["LIGHTLLM_TRITON_AUTOTUNE_LEVEL"] = str(value)
    get_triton_autotune_level.cache_clear()


def init_tune_distributed_if_needed(args):
    if not _is_torchrun():
        return False

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", torch.cuda.device_count()))
    node_rank = args.node_rank if args.node_rank is not None else int(os.environ.get("GROUP_RANK", "0"))
    nnodes = args.nnodes

    if world_size != args.tp:
        raise ValueError(
            f"torchrun WORLD_SIZE must match LightLLM --tp for distributed tune, got WORLD_SIZE={world_size}, "
            f"--tp={args.tp}. Use `torchrun --nproc_per_node={args.tp // args.nnodes}` for this single-node slice."
        )
    tp_size_per_dp(args)
    torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        try:
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                rank=rank,
                world_size=world_size,
                device_id=torch.device(f"cuda:{local_rank}"),
            )
        except TypeError:
            dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)

    from lightllm.distributed.communication_op import dist_group_manager
    from lightllm.utils.dist_utils import (
        set_current_device_id,
        set_current_rank_in_dp,
        set_current_rank_in_node,
        set_dp_rank_in_node,
        set_dp_size,
        set_dp_world_size,
        set_global_dp_rank,
        set_global_rank,
        set_global_world_size,
        set_node_world_size,
    )
    from lightllm.utils.envs_utils import set_env_start_args

    set_global_rank(rank)
    set_global_world_size(world_size)
    set_dp_size(args.dp)
    set_dp_world_size(world_size // args.dp)
    set_global_dp_rank(rank // (world_size // args.dp))
    set_current_rank_in_dp(rank % (world_size // args.dp))
    set_current_rank_in_node(local_rank)
    set_node_world_size(world_size // nnodes)
    set_dp_rank_in_node((rank // (world_size // args.dp)) % max(1, args.dp // nnodes))
    set_current_device_id(local_rank)

    set_env_start_args(
        {
            "run_mode": "normal",
            "tp": args.tp,
            "dp": args.dp,
            "nnodes": nnodes,
            "node_rank": node_rank,
            "data_type": args.dtype,
            "enable_dp_prefill_balance": False,
            "disable_custom_allreduce": True,
            "enable_custom_allgather": False,
            "use_config_server_to_init_nccl": False,
        }
    )

    if len(dist_group_manager) == 0:
        dist_group_manager.create_groups(group_size=1)

    _log_rank0(
        f"[dist] torchrun world_size={world_size}, local_world_size={local_world_size}, "
        f"dp={args.dp}, tp_per_dp={world_size // args.dp}"
    )
    return True


def tune_lightllm(args, cfg, dtype):
    old_autotune_level = os.environ.get("LIGHTLLM_TRITON_AUTOTUNE_LEVEL")
    _set_triton_autotune_level(args.tune_level)
    tp_per_dp = tp_size_per_dp(args)

    _log_rank0(f"[config] thinker MoE: {cfg}")
    _log_rank0(f"[config] tp={args.tp} dp={args.dp} tp_per_dp={tp_per_dp} quant={args.quant_mode} dtype={args.dtype}")
    _log_rank0(f"[config] tune batch_sizes (num_tokens) = {args.batch_sizes}")
    _log_rank0(f"[tune] LIGHTLLM_TRITON_AUTOTUNE_LEVEL={args.tune_level}")
    _log_rank0("[tune] provider: lightllm\n")

    Autotuner.start_autotune_warmup()
    try:
        for num_tokens in args.batch_sizes:
            _log_rank0(f"[tune] tokens={num_tokens} start")
            torch.cuda.synchronize()
            torch.cuda.manual_seed_all(0)

            inputs = build_inputs(num_tokens, cfg, tp_per_dp, dtype, args.quant_mode)
            shared = compute_shared_topk(inputs, cfg)
            prepared = prepare_lightllm(inputs, cfg, args.quant_mode, shared)

            try:
                run_lightllm(inputs, cfg, args.quant_mode, prepared)
                torch.cuda.synchronize()
                _log_rank0(f"[tune] tokens={num_tokens} done")
            finally:
                del inputs, prepared, shared
                gc.collect()
                torch.cuda.empty_cache()
            if dist.is_initialized():
                dist.barrier()
    finally:
        Autotuner.end_autotune_warmup()
        _set_triton_autotune_level(old_autotune_level)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", "--model_dir", dest="model_dir", default=DEFAULT_MODEL_DIR)
    # For MoE the only workload axis is num_tokens (the GEMM's M); we follow
    # vLLM's benchmark_moe.py and call it --batch-sizes even though it's
    # really "tokens per forward". Default list matches vLLM's tuner sweep.
    ap.add_argument(
        "--batch-sizes",
        type=parse_int_list,
        default=[
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
    ap.add_argument("--dp", type=int, default=1)
    ap.add_argument("--nnodes", type=int, default=1)
    ap.add_argument("--node-rank", "--node_rank", dest="node_rank", type=int, default=None)
    ap.add_argument(
        "--quant-mode",
        "--quant_type",
        "--quant-type",
        dest="quant_mode",
        type=parse_quant_mode,
        default="per-channel",
        help="per-channel matches --quant_type vllm-fp8w8a8; block128 matches vllm-fp8w8a8-b128",
    )
    ap.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument(
        "--tune",
        action="store_true",
        default=False,
        help=(
            "Run LightLLM fused_experts under Autotuner warmup for the requested "
            "batch sizes, then exit. This writes the same autotune cache files "
            "as service warmup without launching a service."
        ),
    )
    ap.add_argument(
        "--tune-level",
        type=int,
        choices=[1, 2],
        default=2,
        help=(
            "LIGHTLLM_TRITON_AUTOTUNE_LEVEL used by --tune: 1 tunes only missing "
            "run keys, 2 force-retunes and overwrites cached run keys."
        ),
    )
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

    tune_distributed = init_tune_distributed_if_needed(args) if args.tune else False
    torch.cuda.manual_seed_all(0)

    cfg = load_thinker_moe_config(args.model_dir)
    tp_per_dp = tp_size_per_dp(args)
    cfg["inter_per_rank"] = cfg["moe_intermediate_size"] // tp_per_dp
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    if args.tune:
        try:
            tune_lightllm(args, cfg, dtype)
        finally:
            if tune_distributed and dist.is_initialized():
                dist.destroy_process_group()
        return

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
    print(f"[config] tp={args.tp} dp={args.dp} tp_per_dp={tp_per_dp} quant={args.quant_mode} dtype={args.dtype}")
    print(f"[config] sweep: batch_sizes (num_tokens) = {args.batch_sizes}")
    print(f"[config] providers: {providers}\n")
    if args.profile:
        print("[config] torch profiler enabled (one cudagraph replay per provider per batch size)\n")

    # Tag the flashinfer column to make non-native paths visible:
    #   '*'  per-tensor FP8 CUTLASS fallback (--flashinfer-per-tensor-fallback)
    #   '^'  trtllm_fp8_block_scale_moe on SM120 via SM100 PTX -> sm_120 JIT
    #        (block128 mode without --flashinfer-per-tensor-fallback)
    flashinfer_starred = (
        "flashinfer" in providers
        and args.flashinfer_per_tensor_fallback
        and args.quant_mode in ("per-channel", "block128")
    )
    flashinfer_carret = (
        "flashinfer" in providers and not args.flashinfer_per_tensor_fallback and args.quant_mode == "block128"
    )

    def _flashinfer_label():
        if flashinfer_starred:
            return "flashinfer*"
        if flashinfer_carret:
            return "flashinfer^"
        return "flashinfer"

    column_labels = {p: (_flashinfer_label() if p == "flashinfer" else p) for p in providers}
    prov_col = "".join(f"{column_labels[p] + '(ms)':>15}" for p in providers)
    header = f"{'tokens':>8} |{prov_col}"
    print(header)
    print("-" * len(header))
    if flashinfer_starred:
        print("  * flashinfer runs per-tensor FP8 CUTLASS (not per-channel / block128); not directly comparable.")
    if flashinfer_carret:
        print(
            "  ^ flashinfer runs trtllm_fp8_block_scale_moe via SM100 PTX -> sm_120 driver JIT "
            "(no native SM120 cubin); perf/correctness not vetted by FlashInfer CI."
        )

    results: Dict = {}
    for num_tokens in args.batch_sizes:
        # Reset philox state left dirty by triton.testing.do_bench_cudagraph.
        # (Note: build_inputs / compute_shared_topk use their own explicit
        # Generator, so this seed reset is cosmetic -- kept only to be
        # defensive against future callers that touch the default state.)
        torch.cuda.synchronize()
        torch.cuda.manual_seed_all(0)
        inputs = build_inputs(num_tokens, cfg, tp_per_dp, dtype, args.quant_mode)
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
                ms, no_cudagraph = bench_provider(
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
                # Mark fall-back-to-plain-do_bench numbers with a trailing '~'
                # (e.g. flashinfer's autotuner breaks cudagraph capture).
                cell = f"{ms:.3f}" + ("~" if no_cudagraph else "")
                row_cells.append(f"{cell:>15}")
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
