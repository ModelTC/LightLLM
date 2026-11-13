"""FP8/E8M0 specialization of packed sliding-window storage."""

from dataclasses import dataclass, field
from .packed_swa_state import PackedSlidingWindowConfig, PackedSlidingWindowState


@dataclass(frozen=True)
class FP8E8M0SlidingWindowConfig(PackedSlidingWindowConfig):
    kv_format: str = field(default="fp8_group64_e8m0", init=False)


class FP8E8M0SlidingWindowState(PackedSlidingWindowState):
    pass
