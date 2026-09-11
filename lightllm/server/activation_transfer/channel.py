"""Readiness and lifetime channel shared by autonomous generation servers."""

from __future__ import annotations

from enum import Enum
from typing import Any, Protocol

from .protocol import (
    ActivationKind,
    ActivationPayload,
    ActivationRef,
    ActivationTransport,
)


class ActivationLeaseState(str, Enum):
    HELD = "held"
    IN_USE = "in_use"
    RELEASED = "released"

    # Compatibility aliases for the previous transfer API.
    SERVER_OWNED = "held"
    CONSUMER_OWNED = "in_use"


class _StoreLease(Protocol):
    def resolve(self) -> ActivationPayload | None: ...

    def release(self) -> None: ...


class ActivationStore(Protocol):
    """One kind/transport implementation behind ActivationChannel."""

    kind: ActivationKind
    transport: ActivationTransport

    def open(self, reference: ActivationRef) -> _StoreLease: ...


class ActivationLease:
    """A consumer's lifetime hold on one producer-owned activation."""

    __slots__ = ("_channel", "_store_lease", "reference", "state")

    def __init__(
        self,
        channel: "ActivationChannel",
        reference: ActivationRef,
        store_lease: _StoreLease,
    ) -> None:
        self._channel = channel
        self._store_lease = store_lease
        self.reference = reference
        self.state = ActivationLeaseState.HELD

    @property
    def owner(self) -> tuple[ActivationKind, int, int]:
        return self.reference.owner

    def mark_in_use(self) -> None:
        """Mark that a model worker may be reading the activation."""

        if self.state is not ActivationLeaseState.HELD:
            raise RuntimeError(
                f"cannot use activation lease in state {self.state.value}: {self.owner}"
            )
        self.state = ActivationLeaseState.IN_USE

    def resolve(self) -> ActivationPayload | None:
        self._require_live()
        return self._store_lease.resolve()

    def release(self) -> None:
        if self.state is ActivationLeaseState.RELEASED:
            return
        self._channel._release(self)

    def _require_live(self) -> None:
        if self.state is ActivationLeaseState.RELEASED:
            raise RuntimeError(f"activation lease was already released: {self.owner}")


class ActivationChannel:
    """Thin store router for acquire/resolve/release.

    Scheduling, batching, cancellation and persistent contexts remain owned by
    the modality-specific generation server.
    """

    def __init__(self, *stores: ActivationStore) -> None:
        self._stores: dict[
            tuple[ActivationKind, ActivationTransport], ActivationStore
        ] = {}
        self._leases: dict[tuple[ActivationKind, int, int], ActivationLease] = {}
        for store in stores:
            self.register_store(store)

    def register_store(self, store: ActivationStore) -> None:
        key = (store.kind, store.transport)
        if key in self._stores:
            raise ValueError(
                f"duplicate activation store: {store.kind.value}/{store.transport.value}"
            )
        self._stores[key] = store

    @classmethod
    def for_shm_kv(
        cls,
        shm_req_manager: Any,
        *,
        expected_ready_mask: int,
    ) -> "ActivationChannel":
        from .shm_kv import ShmKVActivationStore

        return cls(
            ShmKVActivationStore(
                shm_req_manager,
                expected_ready_mask=expected_ready_mask,
            )
        )

    def acquire(self, reference: ActivationRef) -> ActivationLease:
        if not isinstance(reference, ActivationRef):
            raise TypeError("activation channel requires an ActivationRef")
        if reference.owner in self._leases:
            raise ValueError(f"duplicate activation owner: {reference.owner}")
        try:
            store = self._stores[(reference.kind, reference.transport)]
        except KeyError as exc:
            raise RuntimeError(
                "no store registered for activation "
                f"{reference.kind.value}/{reference.transport.value}"
            ) from exc

        store_lease = store.open(reference)
        lease = ActivationLease(self, reference, store_lease)
        self._leases[reference.owner] = lease
        return lease

    def resolve(self, reference: ActivationRef) -> ActivationPayload | None:
        try:
            lease = self._leases[reference.owner]
        except KeyError as exc:
            raise RuntimeError(f"unknown activation owner: {reference.owner}") from exc
        return lease.resolve()

    def release(self, reference: ActivationRef) -> None:
        try:
            lease = self._leases[reference.owner]
        except KeyError as exc:
            raise RuntimeError(
                f"unknown activation owner during release: {reference.owner}"
            ) from exc
        lease.release()

    def _release(self, lease: ActivationLease) -> None:
        current = self._leases.get(lease.owner)
        if current is not lease:
            if lease.state is ActivationLeaseState.RELEASED:
                return
            raise RuntimeError(f"unknown activation lease during release: {lease.owner}")
        self._leases.pop(lease.owner)
        try:
            lease._store_lease.release()
        finally:
            lease.state = ActivationLeaseState.RELEASED

    def release_idle(self) -> None:
        self._release_matching(ActivationLeaseState.HELD)

    def release_all(self) -> None:
        self._release_matching(None)

    def _release_matching(self, state: ActivationLeaseState | None) -> None:
        first_error: BaseException | None = None
        for lease in tuple(self._leases.values()):
            if state is not None and lease.state is not state:
                continue
            try:
                lease.release()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    def __contains__(self, reference_or_owner) -> bool:
        if isinstance(reference_or_owner, ActivationRef):
            owner = reference_or_owner.owner
        elif len(reference_or_owner) == 2:
            owner = (ActivationKind.KV, *reference_or_owner)
        else:
            owner = tuple(reference_or_owner)
        return owner in self._leases

    def __len__(self) -> int:
        return len(self._leases)


__all__ = [
    "ActivationChannel",
    "ActivationLease",
    "ActivationLeaseState",
    "ActivationStore",
]
