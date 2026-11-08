from dataclasses import dataclass
from math import isfinite
from typing import Any, Protocol


class StateManager(Protocol):
    """The narrow state-management seam required by execution storage.

    Execution storage may register native state and query the shared request
    capacity.  Cache admission, eviction and asynchronous retirement remain
    owned by the manager implementation; they are deliberately not exposed
    through this construction interface.
    """

    @property
    def request_capacity(self) -> int:
        ...

    def register_state(self, model_role: str, state: Any, resource_id: str | None = None) -> None:
        ...

    def retire(
        self,
        completion: Any,
        *,
        token_indices: Any = None,
        big_pages: tuple[Any, ...] = (),
        small_pages: tuple[Any, ...] = (),
        keepalive: Any = None,
    ) -> None:
        """Release request/page ownership after asynchronous users drain."""
        ...


class StateStore(Protocol):
    """Native views for one execution state retention class."""

    def views(self) -> dict[tuple[str, str], "StateView"]:
        ...

    def close(self) -> None:
        """Release optional provider-owned resources.

        Existing stores may omit this method; callers use it as an optional
        teardown hook so native stores can become explicit owners over time.
        """
        ...


class StateParams(Protocol):
    def validate(self) -> None:
        ...


@dataclass(frozen=True)
class StateSpec:
    role: str
    resource: str
    producer: int
    consumers: tuple[int, ...]
    params: StateParams
    retention: str = "history"

    def __post_init__(self):
        if not self.role or not self.resource or not self.consumers:
            raise ValueError("state identity and consumers are required")
        if self.producer not in self.consumers or len(set(self.consumers)) != len(self.consumers):
            raise ValueError("state consumers must be unique and include the producer")
        if min(self.consumers) < 0:
            raise ValueError("state needs nonnegative layer IDs")
        self.params.validate()

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


@dataclass(frozen=True)
class LayerStates:
    primary: StateSpec
    auxiliary: tuple[tuple[str, StateSpec], ...] = ()
