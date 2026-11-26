"""Choose the attention implementation for a layer config and phase.

Implementations are named as on the command line (--llm_prefill_att_backend /
--llm_decode_att_backend): "triton", "fa3", "flashinfer" for GQA and MLA, "flashmla_sparse" for
sparse MLA, and, in the second value, "triton", "flashqla" or "flashkda" for linear attention. One
name can cover several backends of a family, such as the plain and quantized-KV Triton GQA backends;
the first whose constructor accepts the layer config is used.
"""

from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Callable, Optional, Sequence

import torch.distributed as dist

from lightllm.utils.log_utils import init_logger
from .backend import AttentionBackend
from .backends.gqa import GQALayerConfig
from .backends.mla import MLADecodeLayerConfig, MLAPrefillLayerConfig
from .backends.sparse_mla import SparseMLALayerConfig, SWASparseMLALayerConfig
from .backends.gdn import GDNLayerConfig
from .backends.kda import KDALayerConfig

logger = init_logger(__name__)

PHASES = ("prefill", "decode")

# Auto-selection order, best first.
DEFAULT_PRIORITY = {
    "prefill": ("fa3", "flashinfer", "triton"),
    "decode": ("flashinfer", "fa3", "triton"),
}


@dataclass(frozen=True)
class Family:
    """The implementations registered for one layer config type."""

    # phase -> implementation name -> backends ("module:Class"), tried in order.
    backends: dict[str, dict[str, tuple[str, ...]]]
    # phase -> auto-selection order, best first.
    priority: dict[str, tuple[str, ...]] = field(default_factory=lambda: dict(DEFAULT_PRIORITY))
    # The value of --llm_{prefill,decode}_att_backend that names its implementation:
    # 0 for full-attention families, 1 for linear-attention families.
    arg_index: int = 0
    # Names "auto" takes without validating the library.
    unvalidated: tuple[str, ...] = ()


_GQA = "lightllm.common.attention.backends.gqa"
_MLA = "lightllm.common.attention.backends.mla"
_SPARSE = "lightllm.common.attention.backends.sparse_mla"
_GDN = "lightllm.common.attention.backends.gdn"
_KDA = "lightllm.common.attention.backends.kda"
# Sparse MLA has one name: each phase runs FlashMLA where it supports the cache format and FA3
# elsewhere, and auto-selection validates FlashMLA's sparse kernels.
_SPARSE_PRIORITY = {"prefill": ("flashmla_sparse",), "decode": ("flashmla_sparse",)}

_REGISTRY: dict[type, Family] = {
    GQALayerConfig: Family(
        {
            "prefill": {
                "triton": (
                    f"{_GQA}.triton:TritonGQAPrefillBackend",
                    f"{_GQA}.triton_quant:TritonQuantGQAPrefillBackend",
                ),
                "fa3": (f"{_GQA}.fa3:FA3GQAPrefillBackend",),
                "flashinfer": (f"{_GQA}.flashinfer:FlashInferGQAPrefillBackend",),
            },
            "decode": {
                "triton": (
                    f"{_GQA}.triton:TritonGQADecodeBackend",
                    f"{_GQA}.triton_quant:TritonQuantGQADecodeBackend",
                ),
                "fa3": (f"{_GQA}.fa3:FA3GQADecodeBackend",),
                "flashinfer": (f"{_GQA}.flashinfer:FlashInferGQADecodeBackend",),
            },
        }
    ),
    MLAPrefillLayerConfig: Family(
        {
            "prefill": {
                "triton": (f"{_MLA}.triton:TritonMLAPrefillBackend",),
                "fa3": (f"{_MLA}.fa3:FA3MLAPrefillBackend",),
                "flashinfer": (f"{_MLA}.flashinfer:FlashInferMLAPrefillBackend",),
            }
        }
    ),
    MLADecodeLayerConfig: Family(
        {
            "decode": {
                "triton": (f"{_MLA}.triton:TritonMLADecodeBackend",),
                "fa3": (f"{_MLA}.fa3:FA3MLADecodeBackend",),
                "flashinfer": (f"{_MLA}.flashinfer:FlashInferMLADecodeBackend",),
            }
        }
    ),
    SparseMLALayerConfig: Family(
        {
            "prefill": {
                "flashmla_sparse": (
                    f"{_SPARSE}.flashmla:FlashMLASparsePrefillBackend",
                    f"{_SPARSE}.fa3:FA3SparseMLAPrefillBackend",
                )
            },
            "decode": {
                "flashmla_sparse": (
                    f"{_SPARSE}.fa3:FA3SparseMLADecodeBackend",
                    f"{_SPARSE}.flashmla:FlashMLASparseDecodeBackend",
                )
            },
        },
        priority=_SPARSE_PRIORITY,
    ),
    SWASparseMLALayerConfig: Family(
        {
            "prefill": {"flashmla_sparse": (f"{_SPARSE}.flashmla_swa:FlashMLASWAPrefillBackend",)},
            "decode": {"flashmla_sparse": (f"{_SPARSE}.flashmla_swa:FlashMLASWADecodeBackend",)},
        },
        priority=_SPARSE_PRIORITY,
    ),
    # Linear attention: native or replay state, chosen by the layer config's replay.
    GDNLayerConfig: Family(
        {
            "prefill": {
                "flashqla": (f"{_GDN}.flashqla:FlashQLAGDNPrefillBackend",),
                "triton": (f"{_GDN}.triton:TritonGDNPrefillBackend",),
            },
            "decode": {"triton": (f"{_GDN}.triton:TritonGDNDecodeBackend", f"{_GDN}.triton:ReplayGDNDecodeBackend")},
        },
        priority={"prefill": ("flashqla", "triton"), "decode": ("triton",)},
        arg_index=1,
        unvalidated=("triton",),
    ),
    # FlashKDA is a fast path of the KDA prefill implementations, chosen by the layer config's
    # prefill_kernel; "flashkda" therefore names the same backends as "triton".
    KDALayerConfig: Family(
        {
            "prefill": {
                name: (f"{_KDA}.triton:TritonKDAPrefillBackend", f"{_KDA}.triton:ReplayKDAPrefillBackend")
                for name in ("triton", "flashkda")
            },
            "decode": {"triton": (f"{_KDA}.triton:TritonKDADecodeBackend", f"{_KDA}.triton:ReplayKDADecodeBackend")},
        },
        priority={"prefill": ("triton",), "decode": ("triton",)},
        arg_index=1,
        unvalidated=("triton",),
    ),
}


def _family(layer_config: Any) -> Family:
    family = _REGISTRY.get(type(layer_config))
    if family is None:
        raise NotImplementedError(f"no attention implementations are registered for {type(layer_config).__name__}")
    return family


def requested_backend(args, phase: str, layer_config: Any) -> str:
    """The backend name the command line requests for layer_config's family in phase.

    Full-attention families read the first value of --llm_{prefill,decode}_att_backend and
    linear-attention families the second.
    """
    index = _family(layer_config).arg_index
    values = args.llm_prefill_att_backend if phase == "prefill" else args.llm_decode_att_backend
    return values[index] if index < len(values) else "auto"


def auto_priority(layer_config: Any, phase: str, mtp_step: int = 0) -> tuple[str, ...]:
    """The order in which "auto" tries the implementations of layer_config's family in this phase.

    With MTP, decode tries FA3 first, which makes better use of Tensor Cores.
    """
    priority = _family(layer_config).priority[phase]
    if phase == "decode" and mtp_step > 0 and "fa3" in priority:
        priority = ("fa3",) + tuple(name for name in priority if name != "fa3")
    return priority


def _construction_failures(reason: Optional[str]) -> list[str]:
    """Agree on each candidate before any rank enters the validation broadcast."""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return [reason] if reason else []
    reports = [None] * dist.get_world_size()
    dist.all_gather_object(reports, reason)
    if all(report == reports[0] for report in reports):
        return [reports[0]] if reports[0] else []
    return [f"rank {rank}: {report}" for rank, report in enumerate(reports) if report]


def _construct(
    paths: Sequence[str], layer_config: Any, phase: str, *, agree: bool = False
) -> tuple[Optional[AttentionBackend], list[str]]:
    """Return the first backend in paths ("module:Class") whose constructor accepts layer_config, and why
    the others did not. A backend registered for a phase it does not run in raises TypeError."""
    reasons = []
    for path in paths:
        reason = backend = None
        label = path
        try:
            module_name, class_name = path.split(":")
            backend_cls = getattr(import_module(module_name), class_name)
            label = backend_cls.name
            if backend_cls.phase != phase:
                raise TypeError(f"{label} is a {backend_cls.phase} backend, registered for {phase}")
            backend = backend_cls(layer_config)
        except (NotImplementedError, ImportError, OSError) as error:
            reason = f"{label}: {error}"
        failures = _construction_failures(reason) if agree else ([reason] if reason else [])
        if not failures:
            return backend, reasons
        reasons.extend(failures)
    return None, reasons


def _validate_library(name: str) -> bool:
    from lightllm.utils.backend_validator import validate

    return validate(name)


def select(
    layer_config: Any,
    phase: str,
    requested: str,
    *,
    priority: Optional[Sequence[str]] = None,
    excluded: Sequence[str] = (),
    validate: Callable[[str], bool] = _validate_library,
) -> AttentionBackend:
    """Construct the backend for layers with layer_config in phase ("prefill" or "decode").

    requested is an implementation name or "auto". A named implementation that does not accept
    the layer config raises ValueError. "auto" takes the first name in priority (default
    auto_priority(layer_config, phase)) outside excluded whose constructor accepts the layer config and
    whose library passes validate(name); if none does, it falls back to Triton without validation.
    """
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}, got {phase!r}")
    family = _family(layer_config)
    backends = family.backends.get(phase)
    if backends is None:
        raise ValueError(f"{type(layer_config).__name__} has no {phase} attention implementations")
    if requested != "auto":
        if requested not in backends:
            raise ValueError(
                f"{phase} attention {requested!r} is not available for {layer_config}; choose from {tuple(backends)}"
            )
        backend, reasons = _construct(backends[requested], layer_config, phase)
        if backend is None:
            raise ValueError(f"{phase} attention {requested!r} does not support {layer_config}: {'; '.join(reasons)}")
        return backend

    priority = auto_priority(layer_config, phase) if priority is None else tuple(priority)
    candidates = [name for name in priority if name in backends and name not in excluded]
    reasons = []
    fallback = None
    for name in dict.fromkeys([*candidates, "triton"]):
        backend, failures = _construct(backends.get(name, ()), layer_config, phase, agree=True)
        reasons.extend(failures)
        if backend is None:
            continue
        if name == "triton":
            fallback = backend
        if name not in candidates:
            continue
        if name in family.unvalidated or validate(name):
            logger.info(f"Auto-selected {name} for {phase} attention (validated)")
            return backend
        reasons.append(f"{name}: validation failed")
    if fallback is None:
        raise ValueError(f"no {phase} attention implementation supports {layer_config}: {'; '.join(reasons)}")
    logger.warning(f"No validated {phase} attention implementation supports the layer; falling back to triton")
    return fallback
