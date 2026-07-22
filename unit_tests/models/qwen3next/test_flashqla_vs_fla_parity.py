"""Parity test: FlashQLA dispatch vs FLA Triton fallback.

Drives lightllm.models.qwen3next.triton_kernel.fla.ops.chunk.chunk_gated_delta_rule
twice over the same inputs — once with LIGHTLLM_DISABLE_FLASHQLA=0 (real flash_qla)
and once with LIGHTLLM_DISABLE_FLASHQLA=1 (FLA Triton fallback) — and compares
output tensors `o` and `final_state` within bf16 tolerance.

Tensor construction style is borrowed from FlashQLA tests/test_gdr.py and
benchmark/bench_gated_delta_rule.py (l2norm'd q/k, logsigmoid g/16, sigmoid beta,
fp32 g/beta/h0, bf16 q/k/v).

Benchmark mode (latency comparison, prints a table):
    LIGHTLLM_RUN_BENCH=1 CUDA_VISIBLE_DEVICES=7 pytest -s \\
        unit_tests/models/qwen3next/test_flashqla_vs_fla_parity.py::test_benchmark_flashqla_vs_fla
or:
    CUDA_VISIBLE_DEVICES=7 python unit_tests/models/qwen3next/test_flashqla_vs_fla_parity.py
"""

import gc
import importlib
import math
import os
import sys
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

# When run as `python xxx.py`, sys.path[0] is the script's directory, NOT the
# project root, so `import lightllm` may resolve to an unrelated system-installed
# copy (e.g. /lightllm/) instead of the in-tree one we want to test. Force the
# in-tree project root to the front of sys.path before any lightllm import.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pytest
import torch
import torch.nn.functional as F


# ----------------------- Skips --------------------------------------------- #

SM90_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (9, 0),
    reason="FlashQLA requires SM90+ (Hopper)",
)


def _flashqla_available() -> bool:
    try:
        import flash_qla  # noqa: F401
    except ImportError:
        return False
    return True


FLASHQLA_REQUIRED = pytest.mark.skipif(not _flashqla_available(), reason="flash_qla not installed")


# ----------------------- Helpers ------------------------------------------- #


def _cleanup_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, p=2, dim=-1)


def _make_inputs(
    seqlens: List[int],
    h_qk: int,
    h_v: int,
    head_dim: int,
    use_h0: bool,
    seed: int = 42,
):
    """Build (q, k, v, g, beta, h0, cu_seqlens) following FlashQLA test conventions."""
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(seed)
    num_seqs = len(seqlens)
    total = sum(seqlens)

    offsets = [0]
    for s in seqlens:
        offsets.append(offsets[-1] + s)
    cu_seqlens = torch.tensor(offsets, dtype=torch.int32, device=device)

    q = _l2norm(torch.randn(1, total, h_qk, head_dim, device=device, dtype=dtype))
    k = _l2norm(torch.randn(1, total, h_qk, head_dim, device=device, dtype=dtype))
    v = torch.randn(1, total, h_v, head_dim, device=device, dtype=dtype)
    g = F.logsigmoid(torch.randn(1, total, h_v, device=device, dtype=torch.float32)) / 16
    beta = torch.randn(1, total, h_v, device=device, dtype=torch.float32).sigmoid()

    # Mimic Qwen3-Next SWA mask (~75% heads gated, rest fully open)
    swa = torch.zeros(h_v, dtype=torch.bool, device=device)
    swa[: math.ceil(0.75 * h_v)] = True
    swa = swa[torch.randperm(h_v, device=device)]
    g[:, :, ~swa] = 0.0

    h0 = torch.randn(num_seqs, h_v, head_dim, head_dim, device=device, dtype=torch.float32) if use_h0 else None
    return q, k, v, g, beta, h0, cu_seqlens


def _run_chunk(
    *,
    disable_flashqla: bool,
    q,
    k,
    v,
    g,
    beta,
    initial_state,
    cu_seqlens,
    use_qk_l2norm_in_kernel: bool,
):
    """Run lightllm's chunk_gated_delta_rule under a controlled env var.

    Reload the chunk module so its lru_cache picks up the new env var. Returns
    detached fp32 (o, final_state) on CPU for comparison.
    """
    os.environ["LIGHTLLM_DISABLE_FLASHQLA"] = "1" if disable_flashqla else "0"
    from lightllm.models.qwen3next.triton_kernel.fla.ops import chunk as chunk_mod

    importlib.reload(chunk_mod)
    chunk_mod._flashqla_chunk_gated_delta_rule.cache_clear()

    selected = chunk_mod._flashqla_chunk_gated_delta_rule()
    if disable_flashqla:
        assert selected is None, "env var should disable FlashQLA"
    else:
        assert selected is not None, "FlashQLA should be selected on this hardware"

    o, final_state = chunk_mod.chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        head_first=False,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    o_cpu = o.detach().to(torch.float32).cpu()
    fs_cpu = final_state.detach().to(torch.float32).cpu() if final_state is not None else None
    chunk_mod._flashqla_chunk_gated_delta_rule.cache_clear()
    return o_cpu, fs_cpu


def _assert_close(name: str, a: torch.Tensor, b: torch.Tensor, rel_tol: float):
    """FlashQLA-style relative-tolerance check: max-abs diff / max-abs ref."""
    if a is None and b is None:
        return
    assert a is not None and b is not None, f"{name}: one tensor is None, the other isn't"
    assert a.shape == b.shape, f"{name}: shape mismatch {a.shape} vs {b.shape}"
    ref_max = b.abs().max().item()
    diff_max = (a - b).abs().max().item()
    rel = diff_max / max(ref_max, 1e-6)
    assert rel <= rel_tol, (
        f"{name}: max-abs diff {diff_max:.4g} / max-abs ref {ref_max:.4g} " f"= rel {rel:.4g} > tol {rel_tol:.4g}"
    )


# ----------------------- Configs ------------------------------------------- #

# Qwen3-Next-style heads. (h_qk, h_v) — must satisfy h_v % h_qk == 0 (FlashQLA limit).
# Sample a few TP slices.
HEAD_CONFIGS = [
    pytest.param(2, 8, id="h_qk2_h_v8"),  # ~TP8
    pytest.param(4, 16, id="h_qk4_h_v16"),  # ~TP4
    pytest.param(16, 32, id="h_qk16_h_v32"),  # qwen3.5/9B/4B TP1
]

SEQLEN_CONFIGS = [
    pytest.param([1024], id="single_1k"),
    pytest.param([2048, 1024, 512], id="varlen_3seq"),
    pytest.param([4096, 4096], id="even_2x4k"),
]

HEAD_DIM = 128


# ----------------------- Tests --------------------------------------------- #


@SM90_REQUIRED
@FLASHQLA_REQUIRED
@pytest.mark.parametrize("h_qk, h_v", HEAD_CONFIGS)
@pytest.mark.parametrize("seqlens", SEQLEN_CONFIGS)
@pytest.mark.parametrize("use_h0", [False, True], ids=["no_h0", "with_h0"])
def test_flashqla_vs_fla_parity(h_qk, h_v, seqlens, use_h0):
    """Two backends, identical inputs → outputs should match within bf16 tol.

    Mirrors qwen3next's actual call site (use_qk_l2norm_in_kernel=True, cu_seqlens
    set, head_first=False). Tolerance follows FlashQLA's own test_gdr.py: 2% rel.
    """
    _cleanup_cuda()
    q, k, v, g, beta, h0, cu_seqlens = _make_inputs(
        seqlens=seqlens,
        h_qk=h_qk,
        h_v=h_v,
        head_dim=HEAD_DIM,
        use_h0=use_h0,
    )

    o_qla, fs_qla = _run_chunk(
        disable_flashqla=False,
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=h0,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
    )
    _cleanup_cuda()

    o_fla, fs_fla = _run_chunk(
        disable_flashqla=True,
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=h0,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
    )
    _cleanup_cuda()

    # FlashQLA's own test uses 2% rel; bf16 matmul accumulation can drift to ~1-2%.
    _assert_close("o", o_qla, o_fla, rel_tol=0.02)
    _assert_close("final_state", fs_qla, fs_fla, rel_tol=0.02)


@SM90_REQUIRED
@FLASHQLA_REQUIRED
def test_flashqla_vs_fla_parity_no_l2norm():
    """Same parity check but with use_qk_l2norm_in_kernel=False.

    Inputs are pre-normalized so the kernel sees normalized q/k either way; this
    isolates dispatch path differences from the in-kernel L2 normalization step.
    """
    _cleanup_cuda()
    q, k, v, g, beta, h0, cu_seqlens = _make_inputs(
        seqlens=[2048, 1024],
        h_qk=4,
        h_v=16,
        head_dim=HEAD_DIM,
        use_h0=True,
        seed=123,
    )

    o_qla, fs_qla = _run_chunk(
        disable_flashqla=False,
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=h0,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=False,
    )
    _cleanup_cuda()

    o_fla, fs_fla = _run_chunk(
        disable_flashqla=True,
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=h0,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=False,
    )
    _cleanup_cuda()

    _assert_close("o", o_qla, o_fla, rel_tol=0.02)
    _assert_close("final_state", fs_qla, fs_fla, rel_tol=0.02)


# ===========================================================================
# Benchmark: latency comparison FlashQLA vs FLA Triton (forward only)
# ===========================================================================
#
# Style follows FlashQLA/benchmark/bench_gated_delta_rule.py:
#   - tilelang.profiler.do_bench when available (fall back to CUDA Event)
#   - ModelConfig / SeqLenConfig dataclasses
#   - speedup column relative to FlashQLA (fla_ms / qla_ms)
#
# Skipped by default; opt-in via LIGHTLLM_RUN_BENCH=1.

BENCH_ENABLED = os.environ.get("LIGHTLLM_RUN_BENCH", "0").lower() in ("1", "true", "yes")
BENCH_REQUIRED = pytest.mark.skipif(
    not BENCH_ENABLED,
    reason="set LIGHTLLM_RUN_BENCH=1 to run benchmark",
)


@dataclass
class ModelConfig:
    label: str
    h_qk: int
    h_v: int


@dataclass
class SeqLenConfig:
    label: str
    seqlens: List[int]


# Subset borrowed from FlashQLA/benchmark/bench_gated_delta_rule.py FWD configs.
# Pruned to keep total runtime reasonable for an in-tree unit-test bench.
# 397B / 122B-A10B share the same linear-attention head spec across TP slices.
BENCH_MODELS: List[ModelConfig] = [
    ModelConfig("27B TP1", 16, 48),
    ModelConfig("27B TP2", 8, 24),
    ModelConfig("9B/4B TP1", 16, 32),
    ModelConfig("397B/122B TP1", 16, 64),
    ModelConfig("397B/122B TP2", 8, 32),
    ModelConfig("397B/122B TP4", 4, 16),
    ModelConfig("397B/122B TP8", 2, 8),
]

BENCH_SEQLENS: List[SeqLenConfig] = [
    SeqLenConfig("1x4096", [4096]),
    SeqLenConfig("1x8192", [8192]),
    SeqLenConfig("1x16384", [16384]),
    SeqLenConfig("4096+4096", [4096, 4096]),
    SeqLenConfig("8192+8192", [8192, 8192]),
]


def _do_bench(fn: Callable[[], None], warmup: int = 10, rep: int = 50) -> float:
    """Return mean per-call latency in milliseconds.

    Prefer tilelang.profiler.do_bench (same tool FlashQLA's bench uses) for
    parity. Fall back to CUDA Event timing if tilelang is unavailable.
    """
    try:
        import tilelang.profiler

        return float(tilelang.profiler.do_bench(fn, warmup=warmup, rep=rep))
    except Exception:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(rep):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / rep


def _make_chunk_call(*, disable_flashqla: bool, q, k, v, g, beta, initial_state, cu_seqlens) -> Callable[[], None]:
    """Switch the dispatch via env var + cache_clear, return a zero-arg call.

    No importlib.reload needed: `_flashqla_chunk_gated_delta_rule` re-reads the
    env var on each call after `cache_clear()`.
    """
    os.environ["LIGHTLLM_DISABLE_FLASHQLA"] = "1" if disable_flashqla else "0"
    from lightllm.models.qwen3next.triton_kernel.fla.ops import chunk as chunk_mod

    chunk_mod._flashqla_chunk_gated_delta_rule.cache_clear()

    sel = chunk_mod._flashqla_chunk_gated_delta_rule()
    if disable_flashqla:
        assert sel is None, "env var should disable FlashQLA"
    else:
        assert sel is not None, "FlashQLA should be selected on this hardware"

    def call():
        chunk_mod.chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            head_first=False,
            use_qk_l2norm_in_kernel=True,
        )

    return call


BENCH_HEAD_DIM = 128

BENCH_HDR = (
    f"{'Model':<14} {'Seqlens':<14} {'h_qk':>4} {'h_v':>4}    " f"{'flash_qla':>10}  {'fla':>10}    {'speedup':>8}"
)


def _fmt_ms(x: float) -> str:
    if math.isnan(x):
        return "      N/A "
    return f"{x:>8.3f}ms"


def _fmt_speedup(qla: float, fla: float) -> str:
    if math.isnan(qla) or math.isnan(fla) or qla == 0:
        return "    N/A "
    return f"{fla / qla:>6.2f}x"


def _bench_one(seqlens: List[int], h_qk: int, h_v: int, head_dim: int = BENCH_HEAD_DIM) -> Tuple[float, float]:
    """Returns (qla_ms, fla_ms). NaN on per-backend failure."""
    _cleanup_cuda()
    q, k, v, g, beta, h0, cu_seqlens = _make_inputs(
        seqlens=seqlens,
        h_qk=h_qk,
        h_v=h_v,
        head_dim=head_dim,
        use_h0=True,
        seed=42,
    )

    try:
        qla_call = _make_chunk_call(
            disable_flashqla=False,
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=h0,
            cu_seqlens=cu_seqlens,
        )
        qla_ms = _do_bench(qla_call)
    except Exception as e:
        print(f"  [WARN] flash_qla failed: {e}")
        qla_ms = float("nan")
    _cleanup_cuda()

    try:
        fla_call = _make_chunk_call(
            disable_flashqla=True,
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=h0,
            cu_seqlens=cu_seqlens,
        )
        fla_ms = _do_bench(fla_call)
    except Exception as e:
        print(f"  [WARN] fla failed: {e}")
        fla_ms = float("nan")
    _cleanup_cuda()

    return qla_ms, fla_ms


def _run_bench_table(file=None):
    """Print a results table to `file` (default stdout)."""
    print(file=file)
    print(BENCH_HDR, file=file)
    print("-" * len(BENCH_HDR), file=file)

    prev_label = None
    for mc in BENCH_MODELS:
        if prev_label is not None and mc.label != prev_label:
            print(file=file)
        prev_label = mc.label
        for sc in BENCH_SEQLENS:
            qla_ms, fla_ms = _bench_one(sc.seqlens, mc.h_qk, mc.h_v)
            print(
                f"{mc.label:<14} {sc.label:<14} {mc.h_qk:>4} {mc.h_v:>4}    "
                f"{_fmt_ms(qla_ms)}  {_fmt_ms(fla_ms)}    {_fmt_speedup(qla_ms, fla_ms)}",
                file=file,
                flush=True,
            )


@SM90_REQUIRED
@FLASHQLA_REQUIRED
@BENCH_REQUIRED
def test_benchmark_flashqla_vs_fla(capsys):
    """Forward-pass latency comparison: FlashQLA vs FLA Triton fallback.

    Opt in with LIGHTLLM_RUN_BENCH=1. Output goes to stdout (use `pytest -s`).
    """
    with capsys.disabled():
        gpu_name = torch.cuda.get_device_properties(0).name
        print(f"\nGPU: {gpu_name}    head_dim={BENCH_HEAD_DIM}")
        print(
            f"flash_qla={'available' if _flashqla_available() else 'missing'}    "
            f"torch={torch.__version__}    cuda={torch.version.cuda}"
        )
        _run_bench_table()


if __name__ == "__main__":
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (9, 0):
        raise SystemExit("FlashQLA benchmark requires SM90+ GPU")
    if not _flashqla_available():
        raise SystemExit("flash_qla is not installed")

    gpu_name = torch.cuda.get_device_properties(0).name
    print(f"GPU: {gpu_name}    head_dim={BENCH_HEAD_DIM}")
    print(f"flash_qla=available    torch={torch.__version__}    cuda={torch.version.cuda}")
    _run_bench_table()
