"""Choose the attention implementation for a layer config and phase.

Implementations are named as on the command line (--llm_prefill_att_backend /
--llm_decode_att_backend): "triton", "fa3", "flashinfer" for GQA and MLA, "flashmla_sparse" for
sparse MLA. One name can cover several backends of a family, such as the plain and quantized-KV
Triton GQA backends; the first whose impl accepts the layer config is used.
"""

from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Callable, Optional, Sequence

from lightllm.utils.log_utils import init_logger
from .backend import AttentionBackend, AttentionImpl
from .backends.gqa import GQALayerConfig
from .backends.mla import MLADecodeLayerConfig, MLAPrefillLayerConfig
from .backends.sparse_mla import SparseMLALayerConfig, SWASparseMLALayerConfig

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


_GQA = "lightllm.common.attention.backends.gqa"
_MLA = "lightllm.common.attention.backends.mla"
_SPARSE = "lightllm.common.attention.backends.sparse_mla"
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
}


@dataclass(frozen=True)
class Selection:
    backend: type[AttentionBackend]
    impl: AttentionImpl


def _load(path: str) -> type[AttentionBackend]:
    module_name, class_name = path.split(":")
    return getattr(import_module(module_name), class_name)


def _family(layer_config: Any) -> Family:
    family = _REGISTRY.get(type(layer_config))
    if family is None:
        raise NotImplementedError(f"no attention implementations are registered for {type(layer_config).__name__}")
    return family


def _phase_backends(layer_config: Any, phase: str) -> dict[str, tuple[str, ...]]:
    backends = _family(layer_config).backends.get(phase)
    if backends is None:
        raise ValueError(f"{type(layer_config).__name__} has no {phase} attention implementations")
    return backends


def implementation_names(layer_config: Any, phase: str) -> tuple[str, ...]:
    """Names of the implementations registered for layer_config's family in this phase."""
    return tuple(_phase_backends(layer_config, phase))


def arg_index(layer_config: Any) -> int:
    """Which value of --llm_{prefill,decode}_att_backend names the implementation of layer_config's family."""
    return _family(layer_config).arg_index


def auto_priority(layer_config: Any, phase: str, mtp_step: int = 0) -> tuple[str, ...]:
    """The order in which "auto" tries the implementations of layer_config's family in this phase.

    With MTP, decode tries FA3 first, which makes better use of Tensor Cores.
    """
    priority = _family(layer_config).priority[phase]
    if phase == "decode" and mtp_step > 0 and "fa3" in priority:
        priority = ("fa3",) + tuple(name for name in priority if name != "fa3")
    return priority


def _construct(name: str, layer_config: Any, phase: str) -> tuple[Optional[Selection], list[str]]:
    """Return the first backend under name whose impl accepts layer_config, and why the others did not."""
    reasons = []
    for path in _phase_backends(layer_config, phase).get(name, ()):
        backend = _load(path)
        try:
            return Selection(backend, backend.get_impl_cls()(layer_config)), reasons
        except NotImplementedError as e:
            reasons.append(f"{backend.get_name()}: {e}")
    return None, reasons


def default_validate(name: str) -> bool:
    from lightllm.utils.backend_validator import validate

    return validate(name)


def select(
    layer_config: Any,
    phase: str,
    requested: str,
    *,
    priority: Optional[Sequence[str]] = None,
    excluded: Sequence[str] = (),
    validate: Callable[[str], bool] = default_validate,
) -> Selection:
    """Choose the backend for layers with layer_config in phase ("prefill" or "decode").

    requested is an implementation name or "auto". A named implementation that does not accept
    the layer config raises ValueError. "auto" takes the first name in priority (default
    auto_priority(layer_config, phase)) outside excluded whose impl accepts the layer config and
    whose library passes validate(name); if none does, it falls back to Triton without validation.
    """
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}, got {phase!r}")
    names = implementation_names(layer_config, phase)
    if requested != "auto":
        if requested not in names:
            raise ValueError(
                f"{phase} attention {requested!r} is not available for {layer_config}; choose from {names}"
            )
        selection, reasons = _construct(requested, layer_config, phase)
        if selection is None:
            raise ValueError(f"{phase} attention {requested!r} does not support {layer_config}: {'; '.join(reasons)}")
        return selection

    priority = auto_priority(layer_config, phase) if priority is None else tuple(priority)
    for name in priority:
        if name in excluded or name not in names:
            continue
        selection, _ = _construct(name, layer_config, phase)
        # Validate only libraries that could run the layer, since validation starts a subprocess.
        if selection is not None and validate(name):
            logger.info(f"Auto-selected {name} for {phase} attention (validated)")
            return selection
    selection, reasons = _construct("triton", layer_config, phase)
    if selection is None:
        raise ValueError(f"no {phase} attention implementation supports {layer_config}: {'; '.join(reasons)}")
    logger.warning(f"No validated {phase} attention implementation supports the layer; falling back to triton")
    return selection
