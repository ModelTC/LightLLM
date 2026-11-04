from dataclasses import dataclass
from math import isfinite
from typing import Any, Literal


@dataclass(frozen=True)
class StateSpec:
    role: str
    resource: str
    producer: int
    consumers: tuple[int, ...]
    shape: tuple[int, ...]
    retention: str = "history"

    def __post_init__(self):
        if not self.role or not self.resource or not self.consumers:
            raise ValueError("state identity and consumers are required")
        if self.producer not in self.consumers or len(set(self.consumers)) != len(self.consumers):
            raise ValueError("state consumers must be unique and include the producer")
        if min(self.consumers) < 0 or not self.shape or min(self.shape) <= 0:
            raise ValueError("state needs nonnegative layer IDs and positive logical dimensions")

    @property
    def key(self):
        return self.role, self.resource


@dataclass(frozen=True)
class FormatSpec:
    encoding: str
    dtype: Any
    layout: str
    fields: tuple[str, ...]
    page_size: int
    shard: tuple[int, int] = (0, 1)
    version: int = 1
    calibration: str | None = None
    group_size: int | None = None
    scale_dtype: Any = None
    field_dtypes: tuple[Any, ...] = ()
    candidate_slots: int = 1

    def __post_init__(self):
        rank, size = self.shard
        if not self.encoding or not self.layout or not self.fields or len(set(self.fields)) != len(self.fields):
            raise ValueError("format needs an encoding, layout and unique fields")
        if self.page_size <= 0 or self.version <= 0 or size <= 0 or not 0 <= rank < size:
            raise ValueError("invalid page size, format version or shard")
        if self.group_size is not None and (self.group_size < 2 or self.group_size & (self.group_size - 1)):
            raise ValueError("quantization group size must be a power of two >= 2")
        if self.field_dtypes and len(self.field_dtypes) != len(self.fields):
            raise ValueError("field dtypes must describe every field")
        if self.candidate_slots <= 0:
            raise ValueError("state candidate slots must be positive")


@dataclass(frozen=True)
class StateView:
    state: StateSpec
    format: FormatSpec
    # A tuple prevents field replacement; native tensor shape/stride/offset remain authoritative.
    fields: tuple[Any, ...]

    def __post_init__(self):
        if len(self.fields) != len(self.format.fields):
            raise ValueError("view fields must match its format")
