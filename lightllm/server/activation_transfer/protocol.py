"""Small control-plane types for producer-to-generation activations.

Only immutable references cross service boundaries.  Storage coordinates stay
inside typed views returned by :class:`ActivationChannel`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

from lightllm.server.core.objs.kv_consumer import KVActivation, PrefixKVHandle


class ActivationKind(str, Enum):
    KV = "kv"
    PREFIX_KV = "kv"
    HIDDEN_STATES = "hidden_states"


class ActivationTransport(str, Enum):
    CUDA_IPC = "cuda_ipc"
    CPU_SHM = "cpu_shm"


@dataclass(frozen=True)
class ActivationRef:
    """ABA-safe identity of one producer-owned activation.

    ``slot_index`` identifies a reusable storage slot and ``generation`` is
    the request generation currently occupying it.  A stale reference can
    therefore never acquire data from a later request that reused the slot.
    """

    slot_index: int
    generation: int
    kind: ActivationKind
    transport: ActivationTransport = ActivationTransport.CUDA_IPC
    position_offset: int | None = None

    def __post_init__(self) -> None:
        if self.slot_index < 0 or self.generation < 0:
            raise ValueError("activation owner identity must be non-negative")
        if not isinstance(self.kind, ActivationKind):
            raise TypeError("activation kind must be an ActivationKind")
        if not isinstance(self.transport, ActivationTransport):
            raise TypeError("activation transport must be an ActivationTransport")
        if self.position_offset is not None and self.position_offset <= 0:
            raise ValueError("activation position offset must be positive")

    @classmethod
    def kv(
        cls,
        slot_index: int,
        generation: int,
        *,
        position_offset: int | None = None,
        transport: ActivationTransport = ActivationTransport.CUDA_IPC,
    ) -> "ActivationRef":
        return cls(
            slot_index,
            generation,
            ActivationKind.KV,
            transport,
            position_offset,
        )

    @classmethod
    def hidden_states(
        cls,
        slot_index: int,
        generation: int,
        *,
        transport: ActivationTransport,
    ) -> "ActivationRef":
        return cls(
            slot_index,
            generation,
            ActivationKind.HIDDEN_STATES,
            transport,
        )

    @property
    def owner(self) -> tuple[ActivationKind, int, int]:
        return self.kind, self.slot_index, self.generation


@dataclass(frozen=True)
class HiddenStatesDescriptor:
    """Serializable metadata for a hidden-state tensor transport slot."""

    shape: tuple[int, ...]
    dtype: str
    storage_handle: str | bytes
    strides: tuple[int, ...] | None = None
    device_index: int | None = None
    byte_offset: int = 0
    nbytes: int | None = None
    ready_event_handle: bytes | None = None

    def __post_init__(self) -> None:
        if not self.shape or any(dim <= 0 for dim in self.shape):
            raise ValueError("hidden-state shape must contain positive dimensions")
        if not self.dtype:
            raise ValueError("hidden-state dtype must not be empty")
        if not isinstance(self.storage_handle, (str, bytes)) or not self.storage_handle:
            raise ValueError("hidden-state storage handle must not be empty")
        if self.strides is not None and (
            len(self.strides) != len(self.shape)
            or any(stride < 0 for stride in self.strides)
        ):
            raise ValueError("hidden-state strides must match its shape")
        if self.device_index is not None and self.device_index < 0:
            raise ValueError("hidden-state device index must be non-negative")
        if self.byte_offset < 0:
            raise ValueError("hidden-state byte offset must be non-negative")
        if self.nbytes is not None and self.nbytes <= 0:
            raise ValueError("hidden-state byte length must be positive")


@dataclass(frozen=True)
class HiddenStatesActivation:
    """Readable hidden-state descriptor kept alive by an activation lease.

    The model worker opens the CUDA IPC or CPU SHM handle.  No tensor is sent
    through the generation server's Python control-plane serialization.
    """

    reference: ActivationRef
    descriptor: HiddenStatesDescriptor

    def __post_init__(self) -> None:
        if self.reference.kind is not ActivationKind.HIDDEN_STATES:
            raise ValueError("hidden-state view requires a hidden-state reference")


ActivationPayload: TypeAlias = KVActivation | HiddenStatesActivation

__all__ = [
    "ActivationKind",
    "ActivationPayload",
    "ActivationRef",
    "ActivationTransport",
    "HiddenStatesActivation",
    "HiddenStatesDescriptor",
    "KVActivation",
    "PrefixKVHandle",
]
