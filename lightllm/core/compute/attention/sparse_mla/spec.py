from lightllm.core.formats.kv.spec import SparseMLAKVStateSpec
from dataclasses import dataclass
from math import isfinite
from typing import Any, Literal
from lightllm.core.compute.attention.mla.spec import MLAAttentionSpec


@dataclass(frozen=True)
class SparseMLAAttentionSpec(MLAAttentionSpec):
    index_heads: int = 64
    index_dim: int = 128
    index_topk: int = 2048
    index_quant_block: int = 128
    index_scale_fmt: str | None = None

    def __post_init__(self):
        super().__post_init__()
        if min(self.index_heads, self.index_dim, self.index_topk, self.index_quant_block) <= 0:
            raise ValueError("sparse MLA requires positive indexer dimensions and top-k")

    def validate_state(self, request):
        expected = SparseMLAKVStateSpec(latent_dim=self.latent_dim, rope_dim=self.rope_dim, index_dim=self.index_dim)
        if request.state.params != expected or request.params.spec != self:
            raise ValueError("sparse MLA state or parameters disagree with attention semantics")
