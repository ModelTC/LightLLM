from typing import TYPE_CHECKING

from .base import PrefillQueueStrategy
from .strategies import FCFSStrategy, PromoteShortestPrefillStrategy

if TYPE_CHECKING:
    from ..base_backend import ModeBackend


# 在这里注册新的 PrefillQueueStrategy 子类，即可通过启动参数选择。
PREFILL_QUEUE_STRATEGIES = {
    "default": FCFSStrategy,
    "promote_shortest": PromoteShortestPrefillStrategy,
}


def create_prefill_queue_strategy(backend: "ModeBackend") -> PrefillQueueStrategy:
    name = backend.args.prefill_queue_strategy
    if name not in PREFILL_QUEUE_STRATEGIES:
        raise ValueError(f"Unknown prefill queue strategy {name!r}; choose from {', '.join(PREFILL_QUEUE_STRATEGIES)}")
    return PREFILL_QUEUE_STRATEGIES[name](backend)


__all__ = ["PrefillQueueStrategy", "PREFILL_QUEUE_STRATEGIES", "create_prefill_queue_strategy"]
