"""Attention interfaces and compatibility exports, with backend-owned implementations."""

from lightllm.backends import get_operator_backend
from .base_att import BaseAttBackend, BasePrefillAttState, BaseDecodeAttState, AttControl
from .create_utils import (
    get_prefill_att_backend_class,
    get_decode_att_backend_class,
    get_mla_prefill_att_backend_class,
    get_mla_decode_att_backend_class,
    get_nsa_prefill_att_backend_class,
    get_nsa_decode_att_backend_class,
    get_neo_prefill_att_backend_class,
)

_BACKEND_EXPORTS = {
    "TritonAttBackend": "attention.TritonAttBackend",
    "Int4kvTritonAttBackend": "attention.Int4kvTritonAttBackend",
    "Int8kvTritonAttBackend": "attention.Int8kvTritonAttBackend",
    "MlaTritonAttBackend": "attention.MlaTritonAttBackend",
    "NeoTritonAttBackend": "attention.NeoTritonAttBackend",
    "Fa3AttBackend": "attention.Fa3AttBackend",
    "Fp8Fa3AttBackend": "attention.Fp8Fa3AttBackend",
    "MlaFa3AttBackend": "attention.MlaFa3AttBackend",
    "NeoFa3AttBackend": "attention.NeoFa3AttBackend",
    "Fp8FlashInferAttBackend": "attention.Fp8FlashInferAttBackend",
    "FlashInferAttBackend": "attention.FlashInferAttBackend",
    "MlaFlashInferAttBackend": "attention.MlaFlashInferAttBackend",
    "NsaFlashMlaSparseAttBackend": "attention.NsaFlashMlaSparseAttBackend",
    "NsaFlashMlaFp8SparseAttBackend": "attention.NsaFlashMlaFp8SparseAttBackend",
}
__all__ = [
    "BaseAttBackend",
    "BasePrefillAttState",
    "BaseDecodeAttState",
    "AttControl",
    "TritonAttBackend",
    "Int4kvTritonAttBackend",
    "Int8kvTritonAttBackend",
    "MlaTritonAttBackend",
    "NeoTritonAttBackend",
    "Fa3AttBackend",
    "Fp8Fa3AttBackend",
    "MlaFa3AttBackend",
    "NeoFa3AttBackend",
    "Fp8FlashInferAttBackend",
    "FlashInferAttBackend",
    "MlaFlashInferAttBackend",
    "NsaFlashMlaSparseAttBackend",
    "NsaFlashMlaFp8SparseAttBackend",
    "get_prefill_att_backend_class",
    "get_decode_att_backend_class",
    "get_mla_prefill_att_backend_class",
    "get_mla_decode_att_backend_class",
    "get_nsa_prefill_att_backend_class",
    "get_nsa_decode_att_backend_class",
    "get_neo_prefill_att_backend_class",
]


def __getattr__(name):
    if name not in _BACKEND_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = get_operator_backend().resolve(_BACKEND_EXPORTS[name])
    globals()[name] = value
    return value
