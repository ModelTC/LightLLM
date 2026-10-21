"""Tune one layer's SSM update cycle using the shared autotuner."""

import ast

import torch
import triton

from lightllm.common.triton_utils.autotuner import Autotuner, AutotuneKernelType, autotune
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


def get_configs():
    return [
        {"BV": bv, "num_warps": warps, "num_stages": stages}
        for bv, warps in [(8, 1), (16, 1), (16, 2), (32, 1), (32, 2), (32, 4), (64, 2), (64, 4), (128, 4)]
        for stages in (1, 2, 3)
    ]


def static_key(ssm_backend, mode, q, k, v, a, b, cu_seqlens):
    axis = 1 if cu_seqlens is not None else 0
    _, _, hv, kd, vd = ssm_backend.state.shape
    # Keep filenames short, but distinguish packed projections, TP layouts,
    # state precision, recurrent rule and history capacity.
    key = {
        "mode": mode,
        "v": 6,
        "shape": str((q.shape[-2], hv, ssm_backend.num_key_heads, kd, vd, ssm_backend.verify_width)),
        "dtype": str((q.dtype, ssm_backend.state.dtype)),
        "strides": str((q.stride(axis), k.stride(axis), v.stride(axis), a.stride(0), b.stride(0))),
        "history": ssm_backend.capacity if mode == "replay" else 0,
        "gate": int(ssm_backend.kda),
        "varlen": cu_seqlens is not None,
    }
    if ssm_backend.kda:
        key["lower"] = ssm_backend.gate_lower_bound
    if mode == "replay":
        key["history_dtype"] = str(ssm_backend.keys.dtype)
        if ssm_backend.compute_mode == "parallel":
            key["projection"] = "parallel"
            key["fold"] = ssm_backend.fold_method
            return key
        key["projection"] = (
            ("kda_mma" if ssm_backend.kda else "gdn_mma")
            if ssm_backend.compute_mode == "precompute" and ssm_backend.verify_width > 1
            else "inline"
        )
    return key


def run_key(ssm_backend, q, cu_seqlens):
    tokens = q.shape[1 if cu_seqlens is not None else 0]
    return tokens, cu_seqlens.numel() - 1 if cu_seqlens is not None else tokens


def scratch_size_bytes(ssm_backend, batch, sequences, inputs):
    # State/history have a leading layer axis; cursors are shared across layers.
    layers, slots = ssm_backend.state.shape[:2]
    pool_bytes = sum(
        x.numel() * x.element_size() // (layers if x.ndim > 1 else 1)
        for x in vars(ssm_backend).values()
        if isinstance(x, torch.Tensor)
    )
    input_bytes = sum(
        (1 + sum((size - 1) * stride for size, stride in zip(x.shape, x.stride()))) * x.element_size()
        for x in inputs
        if x.numel()
    )
    return pool_bytes // slots * (batch + 1) + input_bytes + 32 * (batch + sequences + 1)


def rebuild_inputs(ssm_backend, mode, q, k, v, a, b, a_log, bias, cu_seqlens=None, workload=None, run_config=None):
    tokens, sequences = run_key(ssm_backend, q, cu_seqlens)
    batch = min(sequences, triton.cdiv(tokens, ssm_backend.verify_width)) if cu_seqlens is not None else tokens
    if q.is_cuda:
        required = scratch_size_bytes(ssm_backend, batch, sequences, (q, k, v, a, b, a_log, bias))
        free, _ = torch.cuda.mem_get_info(q.device)
        available = free + torch.cuda.memory_reserved(q.device) - torch.cuda.memory_allocated(q.device)
        # Leave half for benchmark intermediates, CUDA Graphs and allocator fragmentation.
        if required > available // 2:
            raise torch.cuda.OutOfMemoryError(
                f"SSM tuning scratch needs {required} bytes, exceeding its {available // 2}-byte budget"
            )
    _, _, hv, kd, vd = ssm_backend.state.shape
    state = torch.empty((1, batch + 1, hv, kd, vd), device=q.device, dtype=ssm_backend.state.dtype).normal_(0, 0.02)
    if mode == "replay":
        scratch = type(ssm_backend)(
            state,
            ssm_backend.capacity,
            ssm_backend.verify_width,
            q.dtype,
            num_key_heads=ssm_backend.num_key_heads,
            kda=ssm_backend.kda,
            gate_lower_bound=ssm_backend.gate_lower_bound,
            compute_mode=ssm_backend.compute_mode,
            fold_method=ssm_backend.fold_method,
        )
    else:
        scratch = type(ssm_backend)(
            state,
            ssm_backend.verify_width,
            q.dtype,
            num_key_heads=ssm_backend.num_key_heads,
            kda=ssm_backend.kda,
            gate_lower_bound=ssm_backend.gate_lower_bound,
        )
    # Benchmark callbacks must not recursively select a configuration.
    scratch._config_is_fixed = True
    inputs = [
        torch.empty_strided(x.shape, x.stride(), device=x.device, dtype=x.dtype).normal_(0, 0.2)
        for x in (q, k, v, a, b, a_log, bias)
    ]
    reqs = torch.cat(
        (
            torch.arange(batch, dtype=torch.int32, device=q.device),
            torch.full((sequences - batch,), batch, dtype=torch.int32, device=q.device),
        )
    )
    cu = (torch.arange(sequences + 1, dtype=torch.int32, device=q.device) * ssm_backend.verify_width).clamp_max(tokens)
    lengths = cu[1 : batch + 1] - cu[:batch]
    accepted = torch.cat((lengths - 1, lengths.new_zeros(1)))
    partial = accepted.clone()
    partial[:-1] = torch.minimum(reqs[:batch] % ssm_backend.verify_width, lengths - 1)
    return (), dict(
        ssm_backend=scratch,
        mode=mode,
        q=inputs[0],
        k=inputs[1],
        v=inputs[2],
        a=inputs[3],
        b=inputs[4],
        a_log=inputs[5],
        bias=inputs[6],
        cu_seqlens=cu if cu_seqlens is not None else None,
        workload=(reqs, accepted, partial),
    )


@autotune(
    kernel_name="ssm_update_cycle:v1",
    configs_gen_func=get_configs,
    static_key_func=static_key,
    run_key_func=run_key,
    run_key_distance_func=lambda a, b: sum(abs(x - y) for x, y in zip(ast.literal_eval(a), ast.literal_eval(b))),
    kernel_type=AutotuneKernelType.DECODE_ATTENTION,
    rebuild_input_func=rebuild_inputs,
    warmup_all_exist_config=False,
)
def select_config(ssm_backend, mode, q, k, v, a, b, a_log, bias, cu_seqlens=None, workload=None, run_config=None):
    """Return a layout; only rebuilt disposable inputs execute the timed cycle."""
    if workload is None:
        return run_config
    ssm_backend.run_config = run_config
    reqs, accepted, partial = workload
    if mode == "replay":
        ssm_backend.cursors.zero_()
        # Include an entire history period and the subsequent fold. Repeating
        # just the first round would benchmark an empty history forever.
        tokens = q.shape[1 if cu_seqlens is not None else 0]
        rounds = ssm_backend.capacity // min(ssm_backend.verify_width, tokens) + 1
    else:
        rounds = 2  # Partial acceptance followed by full acceptance.
    for step in range(rounds):
        ssm_backend.prepare_decode(reqs, cu_seqlens)
        ssm_backend.forward(0, q, k, v, a, b, a_log, bias, reqs, cu_seqlens)
        ssm_backend.accept_updates(reqs, partial if mode != "replay" and step == 0 else accepted)
    return run_config


def resolve_run_config(ssm_backend, mode, q, k, v, a, b, a_log, bias, cu_seqlens):
    default = ssm_backend.run_config or {"BV": 32, "num_warps": 1}
    if ssm_backend._config_is_fixed:
        return default
    key = (tuple(static_key(ssm_backend, mode, q, k, v, a, b, cu_seqlens).items()), run_key(ssm_backend, q, cu_seqlens))
    if key in ssm_backend.batch_configs:
        return ssm_backend.batch_configs[key]
    if not Autotuner.is_kernel_autotune_warmup(AutotuneKernelType.DECODE_ATTENTION):
        return default
    config = select_config(ssm_backend, mode, q, k, v, a, b, a_log, bias, cu_seqlens)
    config = dict(config or default)
    # Each captured graph keeps its own layout, even after other buckets warm up.
    ssm_backend.batch_configs[key] = config
    logger.info(f"SSM {mode} update-cycle config: {config}, warmup batch: {key[1]}")
    return config
