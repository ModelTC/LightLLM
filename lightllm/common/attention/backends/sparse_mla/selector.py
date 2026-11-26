"""按缓存格式选择稀疏 MLA 后端，SWA 使用自己的 FlashMLA 入口。"""

from typing import TYPE_CHECKING

from . import SparseMLAAttConfig, SWASparseMLAAttConfig
from ...base_backend import AttentionBackend
from ..selection import select_backend

if TYPE_CHECKING:
    from lightllm.common.basemodel.basemodel import TpPartBaseModel


BACKEND_CHOICES = {
    "sparse_mla.prefill": ("flashmla_sparse",),
    "sparse_mla.decode": ("flashmla_sparse",),
    "swa_sparse_mla.prefill": ("flashmla_sparse",),
    "swa_sparse_mla.decode": ("flashmla_sparse",),
}


def select_sparse_mla_prefill_backend(
    att_config: "SparseMLAAttConfig",
    requested: str | None = None,
    *,
    model: "TpPartBaseModel",
    logical_layer_ids: tuple[int, ...],
) -> AttentionBackend:
    """按模型配置选择 sparse_mla 的 prefill 后端。"""
    return select_backend(
        att_config=att_config,
        option=_PREFILL_OPTION,
        implementations=_PREFILL_BACKENDS,
        requested=requested,
        config_type=SparseMLAAttConfig,
        priority=("flashmla_sparse",),
        model=model,
        logical_layer_ids=logical_layer_ids,
    )


def select_sparse_mla_decode_backend(
    att_config: "SparseMLAAttConfig",
    requested: str | None = None,
    *,
    model: "TpPartBaseModel",
    logical_layer_ids: tuple[int, ...],
) -> AttentionBackend:
    """按模型配置选择 sparse_mla 的 decode 后端。"""
    return select_backend(
        att_config=att_config,
        option=_DECODE_OPTION,
        implementations=_DECODE_BACKENDS,
        requested=requested,
        config_type=SparseMLAAttConfig,
        priority=("flashmla_sparse",),
        model=model,
        logical_layer_ids=logical_layer_ids,
    )


def select_swa_sparse_mla_prefill_backend(
    att_config: "SWASparseMLAAttConfig",
    requested: str | None = None,
    *,
    model: "TpPartBaseModel",
    logical_layer_ids: tuple[int, ...],
) -> AttentionBackend:
    """按模型配置选择 swa_sparse_mla 的 prefill 后端。"""
    return select_backend(
        att_config=att_config,
        option=_SWA_PREFILL_OPTION,
        implementations=_SWA_PREFILL_BACKENDS,
        requested=requested,
        config_type=SWASparseMLAAttConfig,
        priority=("flashmla_sparse",),
        model=model,
        logical_layer_ids=logical_layer_ids,
    )


def select_swa_sparse_mla_decode_backend(
    att_config: "SWASparseMLAAttConfig",
    requested: str | None = None,
    *,
    model: "TpPartBaseModel",
    logical_layer_ids: tuple[int, ...],
) -> AttentionBackend:
    """按模型配置选择 swa_sparse_mla 的 decode 后端。"""
    return select_backend(
        att_config=att_config,
        option=_SWA_DECODE_OPTION,
        implementations=_SWA_DECODE_BACKENDS,
        requested=requested,
        config_type=SWASparseMLAAttConfig,
        priority=("flashmla_sparse",),
        model=model,
        logical_layer_ids=logical_layer_ids,
    )


_PREFILL_BACKENDS = {
    "flashmla_sparse": (
        f"{__package__}.flashmla:FlashMLASparsePrefillBackend",
        f"{__package__}.fa3:FA3SparseMLAPrefillBackend",
    )
}

_DECODE_BACKENDS = {
    "flashmla_sparse": (
        f"{__package__}.fa3:FA3SparseMLADecodeBackend",
        f"{__package__}.flashmla:FlashMLASparseDecodeBackend",
    )
}

_SWA_PREFILL_BACKENDS = {"flashmla_sparse": (f"{__package__}.flashmla_swa:FlashMLASWAPrefillBackend",)}

_SWA_DECODE_BACKENDS = {"flashmla_sparse": (f"{__package__}.flashmla_swa:FlashMLASWADecodeBackend",)}

_PREFILL_OPTION = "sparse_mla.prefill"

_DECODE_OPTION = "sparse_mla.decode"

_SWA_PREFILL_OPTION = "swa_sparse_mla.prefill"

_SWA_DECODE_OPTION = "swa_sparse_mla.decode"
