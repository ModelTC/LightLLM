from typing import TYPE_CHECKING

from .base import PrefillQueueStrategy
from .strategies import FCFSStrategy, HRRNPrefillStrategy, PromoteShortestPrefillStrategy

if TYPE_CHECKING:
    from ..base_backend import ModeBackend


# 注册新的策略时，同步更新 CLI 和 StartArgs 的可选值。
PREFILL_QUEUE_STRATEGIES = {
    "default": FCFSStrategy,
    "promote_shortest": PromoteShortestPrefillStrategy,
    "hrrn": HRRNPrefillStrategy,
}


def create_prefill_queue_strategy(backend: "ModeBackend") -> PrefillQueueStrategy:
    name = backend.args.prefill_queue_strategy
    if name not in PREFILL_QUEUE_STRATEGIES:
        raise ValueError(f"Unknown prefill queue strategy {name!r}; choose from {', '.join(PREFILL_QUEUE_STRATEGIES)}")
    return PREFILL_QUEUE_STRATEGIES[name](backend)


__all__ = ["PrefillQueueStrategy", "PREFILL_QUEUE_STRATEGIES", "create_prefill_queue_strategy"]
