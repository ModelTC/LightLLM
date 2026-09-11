from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KVActivation:
    """Immutable view of one target-owned KV activation.

    ``(shm_index, request_id)`` is the reusable-slot generation identity used
    by ActivationTransfer. Consumer adapters use ``infer_req_idx`` and
    ``seq_len`` to snapshot the corresponding rank-local page table and K/V
    pool. Downstream model code never consumes these storage coordinates. The separate
    ``position_offset`` is where a following modality starts; multimodal
    prefixes may compress their logical positions without changing KV length.
    """

    shm_index: int
    request_id: int
    infer_req_idx: int
    seq_len: int
    position_offset: int | None = None

    def __post_init__(self):
        if self.shm_index < 0 or self.request_id < 0:
            raise ValueError("prefix owner identity must be non-negative")
        if self.infer_req_idx < 0:
            raise ValueError("prefix inference request index must be non-negative")
        if self.seq_len <= 0:
            raise ValueError("prefix sequence length must be positive")
        if self.position_offset is None:
            object.__setattr__(self, "position_offset", self.seq_len)
        elif self.position_offset <= 0:
            raise ValueError("prefix position offset must be positive")


# Source compatibility while downstream integrations migrate terminology.
PrefixKVHandle = KVActivation


__all__ = ["KVActivation", "PrefixKVHandle"]
