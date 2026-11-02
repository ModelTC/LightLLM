from dataclasses import dataclass

import torch

from .contracts import ResourceId


@dataclass(frozen=True)
class MlaPrefillInput:
    q: torch.Tensor
    k_nope: torch.Tensor
    k_rope: torch.Tensor
    v: torch.Tensor


@dataclass(frozen=True)
class MlaDecodeInput:
    q_nope: torch.Tensor
    q_rope: torch.Tensor


@dataclass(frozen=True)
class BoundMla:
    resource: ResourceId
    cache: torch.Tensor
    softmax_scale: float
    rope_dim: int
    prefill_slot: int = 0
    decode_slot: int = 0

    def prefill(self, inputs: MlaPrefillInput, metadata, alloc_func):
        return metadata[self.prefill_slot].run(inputs, self, alloc_func)

    def decode(self, inputs: MlaDecodeInput, metadata, alloc_func):
        return metadata[self.decode_slot].run(inputs, self, alloc_func)
