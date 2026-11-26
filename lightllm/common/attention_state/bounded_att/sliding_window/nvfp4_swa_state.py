"""NVFP4 shared-KV sliding window; RoPE retains BF16 precision."""

from dataclasses import dataclass, field
from .packed_swa_state import PackedSlidingWindowConfig, PackedSlidingWindowState


@dataclass(frozen=True)
class NVFP4SlidingWindowConfig(PackedSlidingWindowConfig):
    kv_format: str = field(default="nvfp4_block16", init=False)


class NVFP4SlidingWindowState(PackedSlidingWindowState):
    pass
