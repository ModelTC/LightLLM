"""按层配置选择 KDA 后端，FlashKDA 由 prefill_kernel 决定。"""

from typing import TYPE_CHECKING

from . import KDAAttConfig
from ...base_backend import AttentionBackend
from ..selection import select_backend

if TYPE_CHECKING:
    from lightllm.common.basemodel.basemodel import TpPartBaseModel

BACKEND_CHOICES = {"kda.prefill": ("triton", "flashkda"), "kda.decode": ("triton",)}


def select_kda_prefill_backend(
    att_config: KDAAttConfig,
    *,
    model: "TpPartBaseModel",
    logical_layer_ids: tuple[int, ...],
) -> AttentionBackend:
    """选择 native prefill 后端，kernel 由 att_config.prefill_kernel 确定。"""
    if isinstance(att_config, KDAAttConfig) and att_config.replay:
        raise ValueError("select_kda_prefill_backend requires att_config.replay=False")
    return select_backend(
        att_config=att_config,
        option=_PREFILL_OPTION,
        implementations=_NATIVE_PREFILL_BACKENDS,
        requested="auto",
        config_type=KDAAttConfig,
        priority=("triton",),
        unvalidated=("triton",),
        model=model,
        logical_layer_ids=logical_layer_ids,
    )


def select_kda_replay_prefill_backend(
    att_config: KDAAttConfig,
    *,
    model: "TpPartBaseModel",
    logical_layer_ids: tuple[int, ...],
) -> AttentionBackend:
    """选择 replay prefill 后端，kernel 由 att_config.prefill_kernel 确定。"""
    if isinstance(att_config, KDAAttConfig) and not att_config.replay:
        raise ValueError("select_kda_replay_prefill_backend requires att_config.replay=True")
    return select_backend(
        att_config=att_config,
        option=_PREFILL_OPTION,
        implementations=_REPLAY_PREFILL_BACKENDS,
        requested="auto",
        config_type=KDAAttConfig,
        priority=("triton",),
        unvalidated=("triton",),
        model=model,
        logical_layer_ids=logical_layer_ids,
    )


def select_kda_decode_backend(
    att_config: "KDAAttConfig",
    requested: str | None = None,
    *,
    model: "TpPartBaseModel",
    logical_layer_ids: tuple[int, ...],
) -> AttentionBackend:
    """按模型配置选择 native 状态上的 KDA decode 实现。"""
    if isinstance(att_config, KDAAttConfig) and att_config.replay:
        raise ValueError("select_kda_decode_backend requires att_config.replay=False")
    return select_backend(
        att_config=att_config,
        option=_DECODE_OPTION,
        implementations=_NATIVE_DECODE_BACKENDS,
        requested=requested,
        config_type=KDAAttConfig,
        priority=("triton",),
        unvalidated=("triton",),
        model=model,
        logical_layer_ids=logical_layer_ids,
    )


def select_kda_replay_decode_backend(
    att_config: "KDAAttConfig",
    requested: str | None = None,
    *,
    model: "TpPartBaseModel",
    logical_layer_ids: tuple[int, ...],
) -> AttentionBackend:
    """按模型配置选择 replay 状态上的 KDA decode 实现。"""
    if isinstance(att_config, KDAAttConfig) and not att_config.replay:
        raise ValueError("select_kda_replay_decode_backend requires att_config.replay=True")
    return select_backend(
        att_config=att_config,
        option=_DECODE_OPTION,
        implementations=_REPLAY_DECODE_BACKENDS,
        requested=requested,
        config_type=KDAAttConfig,
        priority=("triton",),
        unvalidated=("triton",),
        model=model,
        logical_layer_ids=logical_layer_ids,
    )


_NATIVE_PREFILL_BACKENDS = {"triton": (f"{__package__}.triton:TritonKDAPrefillBackend",)}

_REPLAY_PREFILL_BACKENDS = {"triton": (f"{__package__}.triton:ReplayKDAPrefillBackend",)}

_NATIVE_DECODE_BACKENDS = {"triton": (f"{__package__}.triton:TritonKDADecodeBackend",)}

_REPLAY_DECODE_BACKENDS = {"triton": (f"{__package__}.triton:ReplayKDADecodeBackend",)}

_PREFILL_OPTION = "kda.prefill"

_DECODE_OPTION = "kda.decode"
