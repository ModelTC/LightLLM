from dataclasses import dataclass
from typing import Callable

from .base_att import AttControl
from .contracts import AttentionSpec, KVFormat


@dataclass(frozen=True)
class BoundAttention:
    spec: AttentionSpec
    format: KVFormat
    cache: tuple
    write_kv: Callable
    control: AttControl
    prefill_slot: int
    decode_slot: int

    def prefill(self, q, metadata, alloc_func):
        return metadata[self.prefill_slot].prefill_att(
            q=q, k=self.cache[0], v=self.cache[1], att_control=self.control, alloc_func=alloc_func
        )

    def decode(self, q, metadata, alloc_func):
        return metadata[self.decode_slot].decode_att(
            q=q, k=self.cache[0], v=self.cache[1], att_control=self.control, alloc_func=alloc_func
        )
