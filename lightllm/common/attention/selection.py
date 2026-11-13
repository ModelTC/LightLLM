import torch
import torch.distributed as dist

from lightllm.common.attention.api.backend import UnsupportedAttention
from lightllm.common.attention.api.spec import AttentionFamily, AttentionPhase
from lightllm.common.attention.registry import (
    VISION_VARLEN_PRIORITY,
    backend_candidates,
    backend_priority,
    load_backend_class,
    varlen_attention_ops,
)
from lightllm.platform import get_device_runtime
from lightllm.utils import backend_validator
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class BackendSelectionError(RuntimeError):
    def __init__(self, family, phase, failures):
        self.failures = failures
        details = "; ".join(f"{name}: {reason}" for name, reason in failures.items())
        super().__init__(f"No usable {family.value} {phase.value} attention backend: {details}")


def select_varlen_attention_op(requested="auto", *, device=None, devices=None, dtype=None, head_dim=80):
    if isinstance(dtype, str):
        aliases = {"fp16": "float16", "bf16": "bfloat16", "fp32": "float32"}
        dtype = getattr(torch, aliases.get(dtype, dtype))
    if devices is None:
        # Resolve once before probing; probes receive concrete device ordinals.
        devices = (get_device_runtime(device).device,)
    else:
        devices = tuple(get_device_runtime(d).device for d in devices)
    if not devices:
        raise ValueError("varlen selection requires at least one device")
    ordered = VISION_VARLEN_PRIORITY if requested == "auto" else (requested,)
    failures = {}
    for name in ordered:
        if name not in varlen_attention_ops:
            failures[name] = "unknown varlen attention operator"
            continue
        # 在子进程中检查实际 Q/K/V 路径，避免父进程初始化 CUDA 或误用文本 attention 探针。
        for assigned_device in devices:
            result = backend_validator.probe_varlen(name, device=assigned_device, dtype=dtype, head_dim=head_dim)
            if not result.ok:
                failures[name] = f"{result.scope.value} on {assigned_device}: {result.reason}"
                break
        if name in failures:
            continue
        logger.info("Selected %s varlen attention after %s", name, result.scope.value)
        return name
    raise BackendSelectionError(AttentionFamily.DENSE, AttentionPhase.PREFILL, failures)


def select_attention_backend(
    *,
    specs,
    caps,
    execution,
    requested="auto",
    candidates=None,
    priority=None,
    ssm_state_mode="native",
):
    specs = tuple(specs)
    if not specs:
        raise ValueError("attention selection requires at least one spec")
    family, phase, encoding = specs[0].family, specs[0].phase, specs[0].cache.format
    if any((spec.family, spec.phase, spec.cache.format) != (family, phase, encoding) for spec in specs):
        raise ValueError("attention specs must share family, phase and cache format")
    if candidates is None:
        candidates = backend_candidates(family, caps.platform, ssm_state_mode)
    available = candidates.get(encoding, {})
    if priority is None:
        priority = backend_priority(family, phase, specs)
    ordered = [requested] if requested != "auto" else list(priority)
    failures = {}
    selected_backend = None
    selected_name = None
    for name in ordered:
        if name not in available:
            failures[name] = f"no implementation for cache encoding {encoding}"
            continue
        probe_scopes = set()
        try:
            backend = load_backend_class(available[name])
            backend.check_environment(caps)
            for spec in specs:
                backend.check_support(spec, caps, execution)
            for spec in specs:
                result = backend_validator.probe_backend(
                    available[name], device=str(caps.device), dtype=spec.dtype, spec=spec
                )
                probe_scopes.add(result.scope.value)
                if not result.ok:
                    failures[name] = f"{result.scope.value}: {result.reason}"
                    break
            if name in failures:
                continue
        except UnsupportedAttention as exc:
            failures[name] = str(exc)
            continue
        except Exception as exc:
            # All ranks must reach consensus, including a rank whose provider check failed.
            logger.exception(
                "Unexpected %s attention provider failure: family=%s phase=%s device=%s specs=%r",
                name,
                family.value,
                phase.value,
                caps.device,
                specs,
            )
            failures[name] = f"provider check failed: {type(exc).__name__}: {exc}"
            continue
        logger.info(
            "Selected %s %s attention on %s: %s (%s)",
            family.value,
            phase.value,
            caps.device,
            name,
            "; ".join(sorted(probe_scopes)),
        )
        selected_backend = backend
        selected_name = name
        break

    error = None if selected_backend is not None else BackendSelectionError(family, phase, failures)
    _check_rank_selection(family, phase, selected_name, None if error is None else str(error))
    if error is not None:
        raise error
    return selected_backend


def _check_rank_selection(family, phase, selected_name, failure):
    if dist.is_initialized() and dist.get_world_size() > 1:
        selections = [None] * dist.get_world_size()
        dist.all_gather_object(selections, {"backend": selected_name, "failure": failure})
        selected = {item["backend"] for item in selections if item["backend"] is not None}
        rejected = {f"rank {rank}": item["failure"] for rank, item in enumerate(selections) if item["failure"]}
        if len(selected) > 1:
            rejected["backend selection"] = ", ".join(sorted(selected))
        if rejected:
            raise BackendSelectionError(family, phase, rejected)
