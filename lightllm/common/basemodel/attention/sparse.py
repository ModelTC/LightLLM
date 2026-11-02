from dataclasses import dataclass, field

import torch

from .contracts import ResourceId


@dataclass(frozen=True)
class SparsePrefillInput:
    q: torch.Tensor
    topk_mem_indices: torch.Tensor
    topk_indices: torch.Tensor | None = None
    new_kv: torch.Tensor | None = None


@dataclass(frozen=True)
class SparseDecodeInput:
    q_nope: torch.Tensor
    q_rope: torch.Tensor
    topk_mem_indices: torch.Tensor


@dataclass(frozen=True)
class BoundSparse:
    resource: ResourceId
    cache: torch.Tensor
    softmax_scale: float
    value_dim: int
    rope_dim: int
    prefill_slot: int = 0
    decode_slot: int = 0

    def prefill(self, inputs: SparsePrefillInput, metadata):
        return metadata[self.prefill_slot].run(inputs, self)

    def decode(self, inputs: SparseDecodeInput, metadata):
        return metadata[self.decode_slot].run(inputs, self)


@dataclass(frozen=True)
class CompressedSparseMetadata:
    swa_indices: torch.Tensor
    swa_lengths: torch.Tensor
    extra_indices: torch.Tensor | None = None
    extra_lengths: torch.Tensor | None = None


@dataclass(frozen=True)
class CompressedSparseInput:
    q: torch.Tensor
    selection: CompressedSparseMetadata


@dataclass(frozen=True)
class BoundCompressedSparse:
    resource: ResourceId
    prefill_cache: torch.Tensor
    decode_cache: torch.Tensor
    compressed_cache: torch.Tensor | None
    compress_ratio: int
    value_dim: int
    softmax_scale: float
    attention_sink: torch.Tensor
    prefill_write_cache: torch.Tensor = field(init=False)
    decode_write_cache: torch.Tensor = field(init=False)

    def __post_init__(self):
        for name in ("prefill", "decode"):
            cache = getattr(self, name + "_cache")

            writer_view = cache.as_strided((cache.shape[0], cache.stride(0)), (cache.stride(0), 1))
            object.__setattr__(self, name + "_write_cache", writer_view)

    def prefill(self, inputs: CompressedSparseInput, metadata, alloc_func, out=None):
        return metadata[0].run(inputs, self, alloc_func, out=out)

    def decode(self, inputs: CompressedSparseInput, metadata, alloc_func):
        return metadata[0].run(inputs, self, alloc_func)
