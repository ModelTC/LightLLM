"""Choose the attention implementation for a layer config and phase.

Implementations are named as on the command line, --att_backend FAMILY.PHASE=NAME (for example
gqa.prefill=fa3 or gdn.prefill=flashqla): "triton", "fa3", "flashinfer" for GQA and MLA,
"flashmla_sparse" for sparse MLA, and "triton", "flashqla" or "flashkda" for linear attention. One
name can cover several backends of a family, such as the plain and quantized-KV Triton GQA backends;
the first whose constructor accepts the layer config is used.
"""

from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Callable, Optional, Sequence

import torch.distributed as dist

from lightllm.utils.backend_validator import validate as validate_library
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

    # The family's name in --att_backend keys ("gqa" in gqa.prefill=fa3); layer config types of one
    # algorithm, such as MLA's prefill and decode configs, share it.
    name: str
    # phase -> implementation name -> backends ("module:Class"), tried in order.
    backends: dict[str, dict[str, tuple[str, ...]]]
    # phase -> auto-selection order, best first.
    priority: dict[str, tuple[str, ...]] = field(default_factory=lambda: dict(DEFAULT_PRIORITY))
    # The value of the deprecated --llm_{prefill,decode}_att_backend flags that names its
    # implementation: 0 for full-attention families, 1 for linear-attention families.
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
        "gqa",
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
        },
    ),
    MLAPrefillLayerConfig: Family(
        "mla",
        {
            "prefill": {
                "triton": (f"{_MLA}.triton:TritonMLAPrefillBackend",),
                "fa3": (f"{_MLA}.fa3:FA3MLAPrefillBackend",),
                "flashinfer": (f"{_MLA}.flashinfer:FlashInferMLAPrefillBackend",),
            }
        },
    ),
    MLADecodeLayerConfig: Family(
        "mla",
        {
            "decode": {
                "triton": (f"{_MLA}.triton:TritonMLADecodeBackend",),
                "fa3": (f"{_MLA}.fa3:FA3MLADecodeBackend",),
                "flashinfer": (f"{_MLA}.flashinfer:FlashInferMLADecodeBackend",),
            }
        },
    ),
    SparseMLALayerConfig: Family(
        "sparse_mla",
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
        "sparse_mla",
        {
            "prefill": {"flashmla_sparse": (f"{_SPARSE}.flashmla_swa:FlashMLASWAPrefillBackend",)},
            "decode": {"flashmla_sparse": (f"{_SPARSE}.flashmla_swa:FlashMLASWADecodeBackend",)},
        },
        priority=_SPARSE_PRIORITY,
    ),
    # Linear attention: native or replay state, chosen by the layer config's replay.
    GDNLayerConfig: Family(
        "gdn",
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
        "kda",
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


class BackendSelector:
    """Select and reuse backend instances while registering one model role."""

    def __init__(self, args):
        self.args = args
        self._backends: dict[tuple, AttentionBackend] = {}
        if args.enable_ep_moe:
            logger.info("Expert parallelism with MoE enabled, excluding flashinfer from auto-selection")

    def select(self, layer_config: Any, phase: str, required_backend: Optional[str] = None) -> AttentionBackend:
        requested = required_backend or requested_backend(self.args, phase, layer_config)
        key = (phase, layer_config, requested)
        if key not in self._backends:
            self._backends[key] = select(
                layer_config,
                phase,
                requested,
                priority=auto_priority(layer_config, phase, self.args.mtp_step),
                excluded=("flashinfer",) if self.args.enable_ep_moe else (),
            )
        return self._backends[key]


def _named_family(name: str) -> Family:
    for family in _REGISTRY.values():
        if family.name == name:
            return family
    raise ValueError(f"no attention family is named {name!r}")


def _family(layer_config: Any) -> Family:
    family = _REGISTRY.get(type(layer_config))
    if family is None:
        raise NotImplementedError(f"no attention implementations are registered for {type(layer_config).__name__}")
    return family


def requested_backend(args, phase: str, family: Any) -> str:
    """The implementation the command line requests for a family in phase.

    family is a family name ("gqa", "kda", ...) or a layer config of the family. --att_backend
    FAMILY.PHASE=NAME takes precedence; otherwise the deprecated --llm_{prefill,decode}_att_backend
    flags give full-attention families their first value and linear-attention families their second.
    """
    family = _named_family(family) if isinstance(family, str) else _family(family)
    override = (getattr(args, "att_backend", None) or {}).get(f"{family.name}.{phase}")
    if override is not None:
        return override
    values = (args.llm_prefill_att_backend if phase == "prefill" else args.llm_decode_att_backend) or ()
    return values[family.arg_index] if family.arg_index < len(values) else "auto"


def backend_choices() -> dict[str, tuple[str, ...]]:
    """The names --att_backend accepts for each FAMILY.PHASE key, "auto" first."""
    choices: dict[str, dict[str, None]] = {}
    for family in _REGISTRY.values():
        for phase, implementations in family.backends.items():
            names = choices.setdefault(f"{family.name}.{phase}", {"auto": None})
            names.update(dict.fromkeys(implementations))
    return {key: tuple(names) for key, names in sorted(choices.items())}


def validate_backend_overrides(overrides: dict[str, str]) -> None:
    """Reject --att_backend entries whose key or implementation name the registry does not know."""
    choices = backend_choices()
    for key, name in overrides.items():
        if key not in choices:
            raise ValueError(f"unknown attention backend key {key!r}; choose from {', '.join(choices)}")
        if name not in choices[key]:
            raise ValueError(f"invalid attention backend {name!r} for {key}; choose from {', '.join(choices[key])}")


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


def select(
    layer_config: Any,
    phase: str,
    requested: str,
    *,
    priority: Optional[Sequence[str]] = None,
    excluded: Sequence[str] = (),
    validate: Callable[[str], bool] = validate_library,
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
