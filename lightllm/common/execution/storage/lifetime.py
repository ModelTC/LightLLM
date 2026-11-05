"""Reclaim storage after its final asynchronous user, without choosing eviction."""

from dataclasses import dataclass
from typing import Callable

from lightllm.common.execution.transfer.completion import Completion


@dataclass(eq=False)
class DeferredRelease:
    completion: Completion
    release: Callable[[], None]
    keepalive: object


class DeferredReclaimer:
    def __init__(self):
        self.pending = []

    def defer(self, completion, release, keepalive=None):
        self.pending.append(DeferredRelease(completion, release, keepalive))

    def collect(self):
        # An earlier slow transfer must not prevent independent completed releases.
        ready = [item for item in self.pending if item.completion.query()]
        for item in ready:
            self.pending.remove(item)
            item.release()
        return len(ready)

    def synchronize(self):
        for item in tuple(self.pending):
            item.completion.synchronize()
        return self.collect()
