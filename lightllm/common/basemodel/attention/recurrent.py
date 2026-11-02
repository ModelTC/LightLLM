from dataclasses import dataclass

import torch

from .contracts import ResourceId


@dataclass(frozen=True)
class GDNInput:
    mixed_qkvzba: torch.Tensor


@dataclass(frozen=True)
class KDAInput:
    mixed_qkv: torch.Tensor
    raw_gate: torch.Tensor
    raw_beta: torch.Tensor


@dataclass(frozen=True)
class BoundRecurrent:
    resource: ResourceId
    state: tuple
    weight: object
    prefill_slot: int = 1
    decode_slot: int = 1

    def prefill(self, inputs: GDNInput | KDAInput, metadata):
        return metadata[self.prefill_slot].run(inputs, self)

    def decode(self, inputs: GDNInput | KDAInput, metadata):
        return metadata[self.decode_slot].run(inputs, self)
