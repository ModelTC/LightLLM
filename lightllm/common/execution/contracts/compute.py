from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class ComputeBatch:
    phase: str
    lane: int = 0


@dataclass(frozen=True)
class ComputeResult:
    tensor: Any
    distribution: Literal["partial", "replicated", "sequence_shard"] = "partial"

    def __post_init__(self):
        if self.distribution not in ("partial", "replicated", "sequence_shard"):
            raise ValueError("unknown compute output distribution")

    def finish(self, reduce, expected="replicated"):
        if self.distribution == "partial":
            return reduce(self.tensor)
        if self.distribution != expected:
            raise ValueError(f"compute produced {self.distribution}, expected {expected}")
        return self.tensor
