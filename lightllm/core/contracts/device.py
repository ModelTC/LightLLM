from dataclasses import dataclass
from math import isfinite
from typing import Any, Literal


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
