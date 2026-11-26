"""分别定义 MLA prefill 和 decode 的后端入口。"""

from typing import TYPE_CHECKING

from . import MLAPrefillAttConfig, MLADecodeAttConfig
from ...base_backend import AttentionBackend
from ..selection import select_backend

if TYPE_CHECKING:
    from lightllm.common.basemodel.basemodel import TpPartBaseModel


BACKEND_CHOICES = {"mla.prefill": ("triton", "fa3", "flashinfer"), "mla.decode": ("triton", "fa3", "flashinfer")}


def select_mla_prefill_backend(
    att_config: "MLAPrefillAttConfig",
    requested: str | None = None,
    *,
    model: "TpPartBaseModel",
    logical_layer_ids: tuple[int, ...],
) -> AttentionBackend:
    """按模型配置选择 mla 的 prefill 后端。"""
    if isinstance(att_config, MLADecodeAttConfig):
        raise ValueError(f"{type(att_config).__name__} has no prefill attention implementations")
    return select_backend(
        att_config=att_config,
        option=_PREFILL_OPTION,
        implementations=_PREFILL_BACKENDS,
        requested=requested,
        config_type=MLAPrefillAttConfig,
        priority=("fa3", "flashinfer", "triton"),
        model=model,
        logical_layer_ids=logical_layer_ids,
    )


def select_mla_decode_backend(
    att_config: "MLADecodeAttConfig",
    requested: str | None = None,
    *,
    model: "TpPartBaseModel",
    logical_layer_ids: tuple[int, ...],
) -> AttentionBackend:
    """按模型配置选择 mla 的 decode 后端。"""
    if isinstance(att_config, MLAPrefillAttConfig):
        raise ValueError(f"{type(att_config).__name__} has no decode attention implementations")
    priority = ("fa3", "flashinfer", "triton") if model.args.mtp_step > 0 else ("flashinfer", "fa3", "triton")
    return select_backend(
        att_config=att_config,
        option=_DECODE_OPTION,
        implementations=_DECODE_BACKENDS,
        requested=requested,
        config_type=MLADecodeAttConfig,
        priority=priority,
        model=model,
        logical_layer_ids=logical_layer_ids,
    )


_PREFILL_BACKENDS = {
    "triton": (f"{__package__}.triton:TritonMLAPrefillBackend",),
    "fa3": (f"{__package__}.fa3:FA3MLAPrefillBackend",),
    "flashinfer": (f"{__package__}.flashinfer:FlashInferMLAPrefillBackend",),
}

_DECODE_BACKENDS = {
    "triton": (f"{__package__}.triton:TritonMLADecodeBackend",),
    "fa3": (f"{__package__}.fa3:FA3MLADecodeBackend",),
    "flashinfer": (f"{__package__}.flashinfer:FlashInferMLADecodeBackend",),
}

_PREFILL_OPTION = "mla.prefill"

_DECODE_OPTION = "mla.decode"
