"""LLM attention backends.

Layout: one directory per attention family (the registry key), one module per kernel library.

    base_att.py          backend / state base classes
    registry.py          backend registration and selection
    flashinfer_utils.py  helpers shared by the FlashInfer backends
    dense/               standard multi-head / grouped-query attention
    mla/                 multi-head latent attention (DeepSeek-V2/V3)
    neo/                 dense attention with image-token visibility (Neo)
    nsa/                 native sparse attention (DeepSeek-V3.2/V4, GLM-5)
    linear/              linear attention with recurrent state (GDN, KDA)

Backend modules are imported lazily by the registry, so they must not be imported here.
"""

from .base_att import BaseAttBackend, BasePrefillAttState, BaseDecodeAttState, AttControl
from .base_att import AttBackendSlot, MAIN_SLOT, LINEAR_SLOT
from .registry import (
    PREFILL,
    DECODE,
    ANY_KV_TYPE,
    register_att_backend,
    resolve_att_backend_class,
)
