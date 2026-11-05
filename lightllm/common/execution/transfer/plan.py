"""Explicit page mapping and bound copy operations, independent of cache policy."""

from dataclasses import dataclass
from typing import Callable, Literal
from lightllm.common.execution.contracts.transfer import TransferRequest


@dataclass(frozen=True)
class CopyOperation:
    resource: str
    layout: object
    run: Callable[[], None]


@dataclass(frozen=True)
class CopyPlan:
    direction: Literal["pack", "unpack", "copy"]
    operations: tuple[CopyOperation, ...]
    request: TransferRequest | None = None

    def __post_init__(self):
        if self.direction not in ("pack", "unpack", "copy"):
            raise ValueError(f"unknown copy direction: {self.direction}")

    def run(self):
        for operation in self.operations:
            operation.run()
