"""普通和 KV 量化 GQA 后端的候选与选择顺序。"""

from typing import TYPE_CHECKING

from . import GQAAttConfig
from ...base_backend import AttentionBackend
from ..selection import select_backend

if TYPE_CHECKING:
    from lightllm.common.basemodel.basemodel import TpPartBaseModel


BACKEND_CHOICES = {"gqa.prefill": ("triton", "fa3", "flashinfer"), "gqa.decode": ("triton", "fa3", "flashinfer")}


def select_gqa_prefill_backend(
    att_config: "GQAAttConfig",
    requested: str | None = None,
    *,
    model: "TpPartBaseModel",
    logical_layer_ids: tuple[int, ...],
) -> AttentionBackend:
    """按模型配置选择 gqa 的 prefill 后端。"""
    return select_backend(
        att_config=att_config,
        option=_PREFILL_OPTION,
        implementations=_PREFILL_BACKENDS,
        requested=requested,
        config_type=GQAAttConfig,
        priority=("fa3", "flashinfer", "triton"),
        model=model,
        logical_layer_ids=logical_layer_ids,
    )


def select_gqa_decode_backend(
    att_config: "GQAAttConfig",
    requested: str | None = None,
    *,
    model: "TpPartBaseModel",
    logical_layer_ids: tuple[int, ...],
) -> AttentionBackend:
    """按模型配置选择 gqa 的 decode 后端。"""
    priority = ("fa3", "flashinfer", "triton") if model.args.mtp_step > 0 else ("flashinfer", "fa3", "triton")
    return select_backend(
        att_config=att_config,
        option=_DECODE_OPTION,
        implementations=_DECODE_BACKENDS,
        requested=requested,
        config_type=GQAAttConfig,
        priority=priority,
        model=model,
        logical_layer_ids=logical_layer_ids,
    )


_PREFILL_BACKENDS = {
    "triton": (
        f"{__package__}.triton:TritonGQAPrefillBackend",
        f"{__package__}.triton_quant:TritonQuantGQAPrefillBackend",
    ),
    "fa3": (f"{__package__}.fa3:FA3GQAPrefillBackend",),
    "flashinfer": (f"{__package__}.flashinfer:FlashInferGQAPrefillBackend",),
}

_DECODE_BACKENDS = {
    "triton": (
        f"{__package__}.triton:TritonGQADecodeBackend",
        f"{__package__}.triton_quant:TritonQuantGQADecodeBackend",
    ),
    "fa3": (f"{__package__}.fa3:FA3GQADecodeBackend",),
    "flashinfer": (f"{__package__}.flashinfer:FlashInferGQADecodeBackend",),
}

_PREFILL_OPTION = "gqa.prefill"

_DECODE_OPTION = "gqa.decode"
