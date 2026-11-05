from dataclasses import dataclass


@dataclass(frozen=True)
class DenseKVStateSpec:
    kv_heads: int
    head_dim: int

    def validate(self):
        if min(self.kv_heads, self.head_dim) <= 0:
            raise ValueError("KV heads and head dimension must be positive")


@dataclass(frozen=True)
class MLAKVStateSpec:
    latent_dim: int
    rope_dim: int

    def validate(self):
        if min(self.latent_dim, self.rope_dim) <= 0:
            raise ValueError("MLA latent and rotary dimensions must be positive")


@dataclass(frozen=True)
class SparseMLAKVStateSpec(MLAKVStateSpec):
    index_dim: int

    def validate(self):
        super().validate()
        if self.index_dim <= 0:
            raise ValueError("indexer dimension must be positive")


@dataclass(frozen=True)
class PooledKVStateSpec:
    latent_dim: int
    index_dim: int

    def validate(self):
        if min(self.latent_dim, self.index_dim) <= 0:
            raise ValueError("pooled KV dimensions must be positive")


@dataclass(frozen=True)
class IndexerWindowStateSpec:
    window: int
    stride: int
    index_dim: int

    def validate(self):
        if min(self.window, self.stride, self.index_dim) <= 0:
            raise ValueError("indexer window, stride and dimension must be positive")
