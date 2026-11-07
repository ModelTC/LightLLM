from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ContextManager, Protocol

if TYPE_CHECKING:
    from torch import Tensor, device, dtype


@dataclass(frozen=True)
class DeviceCaps:
    device_type: str
    index: int | None = None
    capability: tuple[int, int] | None = None
    platform: str | None = None
    architecture: str | None = None

    @property
    def is_nvidia(self):
        return self.device_type == "cuda" and self.platform in (None, "cuda", "nvidia")


@dataclass(frozen=True)
class ExecutionLimits:
    workspace_bytes: int
    max_batch_size: int = 1
    max_seq_len: int = 1
    microbatch_count: int = 1
    max_verify_tokens: int = 1
    graph_phases: tuple[str, ...] = ()
    dynamic_verify: bool = False
    tp_world_size: int = 1
    expert_parallel: bool = False
    transfer_paths: tuple[str, ...] = ()

    def __post_init__(self):
        if self.workspace_bytes < 0:
            raise ValueError("workspace budget must be nonnegative")
        if (
            min(
                self.max_batch_size, self.max_seq_len, self.microbatch_count, self.max_verify_tokens, self.tp_world_size
            )
            <= 0
        ):
            raise ValueError("execution capacities must be positive")
        if any(phase not in ("prefill", "decode", "verify") for phase in self.graph_phases):
            raise ValueError("unknown Graph execution phase")


class DeviceEvent(Protocol):
    def synchronize(self) -> None:
        ...

    def query(self) -> bool:
        ...

    def record(self, stream: DeviceStream | None = None) -> None:
        ...

    def wait(self, stream: DeviceStream | None = None) -> None:
        ...

    def elapsed_time(self, end: DeviceEvent) -> float:
        ...


class DeviceStream(Protocol):
    def synchronize(self) -> None:
        ...

    def wait_event(self, event: DeviceEvent) -> None:
        ...

    def wait_stream(self, stream: DeviceStream) -> None:
        ...


class NativeGraph(Protocol):
    def replay(self) -> None:
        ...

    def reset(self) -> None:
        ...


class GraphBinding(Protocol):
    """Own auxiliary streams, native handles and borrowed storage until close completes."""

    def wait_for_update(self) -> None:
        ...

    def before_replay(self) -> None:
        ...

    def after_replay(self) -> None:
        ...

    def close(self) -> None:
        ...


class DeviceBatch(Protocol):
    def to_device(self, device: device) -> DeviceBatch:
        ...


class DeviceRuntime(Protocol):
    caps: DeviceCaps
    device: device
    distributed_backend: str

    def set_device(self, index: int) -> None:
        ...

    def process_group_options(self) -> dict:
        ...

    def memory_info(self) -> tuple[int, int]:
        ...

    def workspace_budget(self, mem_fraction: float) -> int:
        ...

    def allocate_workspace(self, size: int) -> Tensor:
        ...

    def allocate_tensor(self, shape: tuple[int, ...], dtype: dtype) -> Tensor:
        ...

    def borrow_tensor(self, tensor: Tensor) -> Tensor:
        ...

    def decode_warmup_length(self, max_length: int) -> int:
        ...

    def graph_pool(self) -> object:
        ...

    def create_graph(self) -> NativeGraph:
        ...

    def capture_graph(self, graph: NativeGraph, *, pool: object = None, memory_saver: object = None) -> ContextManager:
        ...

    def synchronize(self) -> None:
        ...

    def barrier(self) -> None:
        ...

    def create_event(self, *, enable_timing: bool = False) -> DeviceEvent:
        ...

    def record_event(self) -> DeviceEvent:
        ...

    def create_stream(self) -> DeviceStream:
        ...

    def current_stream(self) -> DeviceStream:
        ...

    def use_stream(self, stream: DeviceStream | None = None) -> ContextManager:
        ...

    def move_batch(self, batch: DeviceBatch) -> DeviceBatch:
        ...

    def empty_cache(self) -> None:
        ...
