"""按状态类型选择 GDN 后端，prefill 优先尝试 FlashQLA。"""

from typing import TYPE_CHECKING

from . import GDNAttConfig
from ...base_backend import AttentionBackend
from ..selection import select_backend

if TYPE_CHECKING:
    from lightllm.common.basemodel.basemodel import TpPartBaseModel


BACKEND_CHOICES = {"gdn.prefill": ("flashqla", "triton"), "gdn.decode": ("triton",)}


def select_gdn_prefill_backend(
    att_config: "GDNAttConfig",
    requested: str | None = None,
    *,
    model: "TpPartBaseModel",
    logical_layer_ids: tuple[int, ...],
) -> AttentionBackend:
    """按模型配置选择 GDN prefill 实现，att_config.dtype 指定计算输入的 dtype。"""
    return select_backend(
        att_config=att_config,
        option=_PREFILL_OPTION,
        implementations=_PREFILL_BACKENDS,
        requested=requested,
        config_type=GDNAttConfig,
        priority=("flashqla", "triton"),
        unvalidated=("triton",),
        model=model,
        logical_layer_ids=logical_layer_ids,
    )


def select_gdn_decode_backend(
    att_config: "GDNAttConfig",
    requested: str | None = None,
    *,
    model: "TpPartBaseModel",
    logical_layer_ids: tuple[int, ...],
) -> AttentionBackend:
    """按模型配置选择 native 状态上的 GDN decode 实现。"""
    if isinstance(att_config, GDNAttConfig) and att_config.replay:
        raise ValueError("select_gdn_decode_backend requires att_config.replay=False")
    return select_backend(
        att_config=att_config,
        option=_DECODE_OPTION,
        implementations=_NATIVE_DECODE_BACKENDS,
        requested=requested,
        config_type=GDNAttConfig,
        priority=("triton",),
        unvalidated=("triton",),
        model=model,
        logical_layer_ids=logical_layer_ids,
    )


def select_gdn_replay_decode_backend(
    att_config: "GDNAttConfig",
    requested: str | None = None,
    *,
    model: "TpPartBaseModel",
    logical_layer_ids: tuple[int, ...],
) -> AttentionBackend:
    """按模型配置选择 replay 状态上的 GDN decode 实现。"""
    if isinstance(att_config, GDNAttConfig) and not att_config.replay:
        raise ValueError("select_gdn_replay_decode_backend requires att_config.replay=True")
    return select_backend(
        att_config=att_config,
        option=_DECODE_OPTION,
        implementations=_REPLAY_DECODE_BACKENDS,
        requested=requested,
        config_type=GDNAttConfig,
        priority=("triton",),
        unvalidated=("triton",),
        model=model,
        logical_layer_ids=logical_layer_ids,
    )


_PREFILL_BACKENDS = {
    "flashqla": (f"{__package__}.flashqla:FlashQLAGDNPrefillBackend",),
    "triton": (f"{__package__}.triton:TritonGDNPrefillBackend",),
}

_NATIVE_DECODE_BACKENDS = {"triton": (f"{__package__}.triton:TritonGDNDecodeBackend",)}

_REPLAY_DECODE_BACKENDS = {"triton": (f"{__package__}.replay:ReplayGDNDecodeBackend",)}

_PREFILL_OPTION = "gdn.prefill"

_DECODE_OPTION = "gdn.decode"
