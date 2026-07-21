from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable, Hashable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from lightllm.server.actionserver.objs import PrefixContextIdentity


class PrefixContextState(str, Enum):
    """Lifecycle of one immutable prefix-context version."""

    BUILDING = "building"
    ACTIVE = "active"
    DRAINING = "draining"
    RETIRED = "retired"
    POISONED = "poisoned"


class PrefixContextError(RuntimeError):
    """Base error for prefix-context lifecycle violations."""


class UnknownPrefixContext(PrefixContextError):
    pass


class InvalidPrefixContextTransition(PrefixContextError):
    pass


class StalePrefixContextVersion(PrefixContextError):
    pass


class DuplicatePrefixTask(PrefixContextError):
    pass


class PrefixContextReleaseError(PrefixContextError):
    pass


@dataclass(frozen=True)
class PrefixTask:
    context: PrefixContextIdentity
    task_id: Hashable
    payload: Any = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.task_id is None:
            raise ValueError("prefix task_id must not be None")
        hash(self.task_id)


@dataclass(frozen=True)
class PrefixTaskPin:
    """Capability proving that one queued task owns the serial task pin."""

    task: PrefixTask
    pin_id: int

    @property
    def context(self) -> PrefixContextIdentity:
        return self.task.context

    @property
    def task_id(self) -> Hashable:
        return self.task.task_id

    @property
    def payload(self) -> Any:
        return self.task.payload


@dataclass(frozen=True)
class PrefixContextSnapshot:
    identity: PrefixContextIdentity
    state: PrefixContextState
    queued_task_ids: tuple[Hashable, ...]
    pinned_task_id: Hashable | None
    close_reason: str | None
    poison_reason: str | None
    release_error: str | None

    @property
    def has_pin(self) -> bool:
        return self.pinned_task_id is not None

    @property
    def accepts_tasks(self) -> bool:
        return self.state is PrefixContextState.ACTIVE

    @property
    def is_terminal(self) -> bool:
        return self.state in {
            PrefixContextState.RETIRED,
            PrefixContextState.POISONED,
        }


@dataclass
class _ContextRecord:
    identity: PrefixContextIdentity
    resource: Any
    base_identity: PrefixContextIdentity | None = None
    state: PrefixContextState = PrefixContextState.BUILDING
    queue: deque[PrefixTask] = field(default_factory=deque)
    known_task_ids: set[Hashable] = field(default_factory=set)
    # Runtime callbacks are exactly-once, but retaining a small tombstone
    # window makes an immediately repeated ACK harmless without growing for
    # the lifetime of a high-frequency control session.
    acked_pin_ids: deque[int] = field(default_factory=lambda: deque(maxlen=64))
    active_pin: PrefixTaskPin | None = None
    close_reason: str | None = None
    poison_reason: str | None = None
    release_error: str | None = None
    release_started: bool = False
    # A provisional replacement temporarily drains its predecessor so no new
    # work can race the ownership decision, while retaining the predecessor's
    # KV as a rollback point.  Retirement may only start after commit removes
    # this hold, or rollback restores the record to ACTIVE.
    retirement_hold: bool = False


@dataclass(frozen=True)
class _ReleaseAction:
    identity: PrefixContextIdentity
    resource: Any


class PrefixContextRegistry:
    """Thread-safe registry for immutable, versioned prefix resources.

    A replacement is built copy-on-write: the old version remains current
    until :meth:`activate` publishes the replacement. Tasks accepted by the
    old version are then drained in FIFO order. A pinned task is released only
    by an explicit safe acknowledgement. An unsafe acknowledgement poisons the
    version and deliberately prevents its resource-release callback.

    ``release_callback`` is the only resource-specific operation. It is
    invoked as ``release_callback(identity, resource)`` without holding the
    registry lock, so this module has no CUDA or allocator dependency.
    """

    def __init__(
        self,
        release_callback: Callable[[PrefixContextIdentity, Any], None],
        *,
        server_epoch: str = "local",
    ) -> None:
        if not callable(release_callback):
            raise TypeError("release_callback must be callable")
        self._release_callback = release_callback
        self._server_epoch = server_epoch
        self._lock = threading.RLock()
        self._records: dict[PrefixContextIdentity, _ContextRecord] = {}
        self._current: dict[Hashable, PrefixContextIdentity] = {}
        self._building: dict[Hashable, PrefixContextIdentity] = {}
        self._provisional_replaces: dict[
            Hashable,
            PrefixContextIdentity,
        ] = {}
        self._poisoned_context_ids: set[Hashable] = set()
        self._next_version: dict[Hashable, int] = {}
        self._next_pin_id = 1

    def begin_build(
        self,
        context_id: Hashable,
        resource: Any,
    ) -> PrefixContextIdentity:
        """Start the first version for ``context_id`` in BUILDING state."""

        hash(context_id)
        with self._lock:
            if context_id in self._poisoned_context_ids:
                raise InvalidPrefixContextTransition(
                    f"prefix context {context_id!r} is poisoned and cannot be reused"
                )
            if context_id in self._building:
                raise InvalidPrefixContextTransition(
                    f"prefix context {context_id!r} already has a building version"
                )
            if context_id in self._current:
                raise InvalidPrefixContextTransition(
                    f"prefix context {context_id!r} is active; use begin_replace"
                )
            return self._new_record_unlocked(
                context_id,
                resource,
                base_identity=None,
            ).identity

    # Short aliases keep call sites readable without introducing a second
    # lifecycle path.
    begin = begin_build

    def begin_replace(
        self,
        expected: PrefixContextIdentity,
        resource: Any,
    ) -> PrefixContextIdentity:
        """Build a replacement while ``expected`` remains ACTIVE/current."""

        with self._lock:
            self._validate_epoch(expected)
            record = self._record_unlocked(expected)
            if record.state is not PrefixContextState.ACTIVE:
                raise InvalidPrefixContextTransition(
                    "only an active prefix context can be replaced"
                )
            if self._current.get(expected.context_id) != expected:
                raise StalePrefixContextVersion(
                    "replacement base is no longer the current context"
                )
            if expected.context_id in self._building:
                raise InvalidPrefixContextTransition(
                    "a replacement is already being built for this context"
                )
            return self._new_record_unlocked(
                expected.context_id,
                resource,
                base_identity=expected,
            ).identity

    replace = begin_replace

    def activate(self, identity: PrefixContextIdentity) -> PrefixContextSnapshot:
        """Atomically publish a built version and drain its predecessor."""

        release_action = None
        with self._lock:
            self._validate_epoch(identity)
            record = self._record_unlocked(identity)
            if record.state is not PrefixContextState.BUILDING:
                raise InvalidPrefixContextTransition(
                    f"cannot activate context in {record.state.value} state"
                )

            current = self._current.get(identity.context_id)
            if record.base_identity is None:
                if current is not None:
                    raise StalePrefixContextVersion(
                        "another prefix-context version became current while building"
                    )
            elif current != record.base_identity:
                raise StalePrefixContextVersion(
                    "replacement base is no longer the current context"
                )

            predecessor = None if current is None else self._record_unlocked(current)
            record.state = PrefixContextState.ACTIVE
            self._current[identity.context_id] = identity
            self._building.pop(identity.context_id, None)

            if predecessor is not None:
                predecessor.state = PrefixContextState.DRAINING
                predecessor.close_reason = "replaced"
                release_action = self._prepare_retire_unlocked(predecessor)

            snapshot = self._snapshot_unlocked(record)

        self._run_release(release_action)
        return snapshot

    def activate_provisional_replace(
        self,
        identity: PrefixContextIdentity,
    ) -> PrefixContextSnapshot:
        """Run a replacement without yet transferring public ownership.

        The replacement becomes ACTIVE so its first action can execute, but its
        predecessor remains the externally visible ``current`` version.  The
        predecessor drains work already accepted before the transaction and is
        held after draining, preserving its KV until :meth:`commit_provisional_replace`
        or :meth:`rollback_provisional_replace` resolves the transaction.
        """

        with self._lock:
            self._validate_epoch(identity)
            record = self._record_unlocked(identity)
            if record.state is not PrefixContextState.BUILDING:
                raise InvalidPrefixContextTransition(
                    f"cannot provisionally activate context in {record.state.value} state"
                )
            if record.base_identity is None:
                raise InvalidPrefixContextTransition(
                    "only a replacement can be provisionally activated"
                )
            if identity.context_id in self._provisional_replaces:
                raise InvalidPrefixContextTransition(
                    "a provisional replacement is already active for this context"
                )

            current = self._current.get(identity.context_id)
            if current != record.base_identity:
                raise StalePrefixContextVersion(
                    "replacement base is no longer the current context"
                )
            predecessor = self._record_unlocked(record.base_identity)
            if predecessor.state is not PrefixContextState.ACTIVE:
                raise StalePrefixContextVersion("replacement base is no longer active")

            predecessor.state = PrefixContextState.DRAINING
            predecessor.close_reason = "provisional_replace"
            predecessor.retirement_hold = True
            record.state = PrefixContextState.ACTIVE
            self._building.pop(identity.context_id, None)
            self._provisional_replaces[identity.context_id] = identity
            return self._snapshot_unlocked(record)

    def is_provisional_replace(self, identity: PrefixContextIdentity) -> bool:
        """Return whether ``identity`` is the unresolved replacement version."""

        with self._lock:
            self._validate_epoch(identity)
            return self._provisional_replaces.get(identity.context_id) == identity

    def commit_provisional_replace(
        self,
        identity: PrefixContextIdentity,
    ) -> PrefixContextSnapshot:
        """Publish a provisionally executed replacement and retire its base."""

        release_action = None
        with self._lock:
            self._validate_epoch(identity)
            record = self._record_unlocked(identity)
            if self._provisional_replaces.get(identity.context_id) != identity:
                raise InvalidPrefixContextTransition(
                    "context is not the active provisional replacement"
                )
            if record.state is not PrefixContextState.ACTIVE:
                raise InvalidPrefixContextTransition(
                    "only an active provisional replacement can be committed"
                )
            assert record.base_identity is not None
            predecessor = self._record_unlocked(record.base_identity)
            if (
                self._current.get(identity.context_id) != predecessor.identity
                or predecessor.state is not PrefixContextState.DRAINING
                or not predecessor.retirement_hold
            ):
                raise StalePrefixContextVersion(
                    "provisional replacement base can no longer be committed"
                )

            self._current[identity.context_id] = identity
            self._provisional_replaces.pop(identity.context_id, None)
            predecessor.retirement_hold = False
            predecessor.close_reason = "replaced"
            release_action = self._prepare_retire_unlocked(predecessor)
            snapshot = self._snapshot_unlocked(record)

        self._run_release(release_action)
        return snapshot

    def rollback_provisional_replace(
        self,
        identity: PrefixContextIdentity,
    ) -> PrefixContextSnapshot:
        """Discard a provisional replacement and restore its predecessor."""

        release_action = None
        with self._lock:
            self._validate_epoch(identity)
            record = self._record_unlocked(identity)
            if self._provisional_replaces.get(identity.context_id) != identity:
                raise InvalidPrefixContextTransition(
                    "context is not the active provisional replacement"
                )
            if record.state is not PrefixContextState.ACTIVE:
                raise InvalidPrefixContextTransition(
                    "only an active provisional replacement can be rolled back"
                )
            assert record.base_identity is not None
            predecessor = self._record_unlocked(record.base_identity)
            if (
                self._current.get(identity.context_id) != predecessor.identity
                or predecessor.state is not PrefixContextState.DRAINING
                or not predecessor.retirement_hold
            ):
                raise StalePrefixContextVersion(
                    "provisional replacement base can no longer be restored"
                )

            self._provisional_replaces.pop(identity.context_id, None)
            predecessor.retirement_hold = False
            predecessor.state = PrefixContextState.ACTIVE
            predecessor.close_reason = None
            self._current[identity.context_id] = predecessor.identity

            record.state = PrefixContextState.DRAINING
            record.close_reason = "provisional_replace_rolled_back"
            release_action = self._prepare_retire_unlocked(record)
            snapshot = self._snapshot_unlocked(predecessor)

        self._run_release(release_action)
        return snapshot

    def enqueue_task(
        self,
        identity: PrefixContextIdentity,
        task_id: Hashable,
        payload: Any = None,
    ) -> PrefixTask:
        """Append one task to an ACTIVE version's FIFO queue."""

        hash(task_id)
        with self._lock:
            self._validate_epoch(identity)
            record = self._record_unlocked(identity)
            if record.state is not PrefixContextState.ACTIVE:
                raise InvalidPrefixContextTransition(
                    f"context in {record.state.value} state does not accept tasks"
                )
            if task_id in record.known_task_ids:
                raise DuplicatePrefixTask(
                    f"task {task_id!r} already exists in prefix context {identity}"
                )
            task = PrefixTask(identity, task_id, payload)
            record.known_task_ids.add(task_id)
            record.queue.append(task)
            return task

    enqueue = enqueue_task

    def pin_next_task(self, identity: PrefixContextIdentity) -> PrefixTaskPin | None:
        """Pin the next queued task, or return ``None`` if busy/empty."""

        with self._lock:
            self._validate_epoch(identity)
            record = self._record_unlocked(identity)
            if record.state not in {
                PrefixContextState.ACTIVE,
                PrefixContextState.DRAINING,
            }:
                raise InvalidPrefixContextTransition(
                    f"context in {record.state.value} state cannot pin tasks"
                )
            if record.active_pin is not None or not record.queue:
                return None
            task = record.queue.popleft()
            pin = PrefixTaskPin(task=task, pin_id=self._next_pin_id)
            self._next_pin_id += 1
            record.active_pin = pin
            return pin

    pin_next = pin_next_task

    def acknowledge_task(self, pin: PrefixTaskPin, *, safe: bool) -> bool:
        """End a pin only after the consumer proves resource safety.

        A repeated safe acknowledgement is idempotent. ``safe=False`` records
        an unreleasable pin and poisons the complete context version.
        """

        release_action = None
        with self._lock:
            self._validate_epoch(pin.context)
            record = self._record_unlocked(pin.context)
            if pin.pin_id in record.acked_pin_ids:
                return False
            if record.active_pin != pin:
                raise InvalidPrefixContextTransition(
                    "task acknowledgement does not own the active context pin"
                )
            if record.state is PrefixContextState.POISONED:
                raise InvalidPrefixContextTransition(
                    "a poisoned prefix context cannot accept a late acknowledgement"
                )

            if not safe:
                record.state = PrefixContextState.POISONED
                record.poison_reason = (
                    f"task {pin.task_id!r} did not provide a safe acknowledgement"
                )
                record.known_task_ids.discard(pin.task_id)
                record.acked_pin_ids.append(pin.pin_id)
                self._poisoned_context_ids.add(pin.context.context_id)
                self._detach_unlocked(record)
                return True

            record.active_pin = None
            record.known_task_ids.discard(pin.task_id)
            record.acked_pin_ids.append(pin.pin_id)
            if record.state is PrefixContextState.DRAINING:
                release_action = self._prepare_retire_unlocked(record)

        self._run_release(release_action)
        return True

    ack = acknowledge_task

    def drain_queued_tasks(
        self, identity: PrefixContextIdentity
    ) -> tuple[PrefixTask, ...]:
        """Remove tasks that can no longer run after a context is poisoned.

        The caller owns completion of the returned payloads.  Keeping this
        explicit prevents queued HTTP requests from disappearing without a
        terminal action result.
        """

        with self._lock:
            self._validate_epoch(identity)
            record = self._record_unlocked(identity)
            if record.state is not PrefixContextState.POISONED:
                raise InvalidPrefixContextTransition(
                    "queued tasks may only be drained from a poisoned context"
                )
            tasks = tuple(record.queue)
            record.queue.clear()
            for task in tasks:
                record.known_task_ids.discard(task.task_id)
            return tasks

    def close(
        self,
        identity: PrefixContextIdentity,
        *,
        reason: str = "closed",
    ) -> bool:
        """Stop accepting work, drain accepted tasks, then release safely."""

        release_action = None
        with self._lock:
            self._validate_epoch(identity)
            record = self._record_unlocked(identity)
            if record.state in {
                PrefixContextState.RETIRED,
                PrefixContextState.POISONED,
                PrefixContextState.DRAINING,
            }:
                return False
            record.state = PrefixContextState.DRAINING
            record.close_reason = reason
            self._detach_unlocked(record)
            release_action = self._prepare_retire_unlocked(record)

        self._run_release(release_action)
        return True

    def poison(
        self,
        identity: PrefixContextIdentity,
        reason: str,
    ) -> bool:
        """Make a version permanently unreleasable without invoking cleanup."""

        with self._lock:
            self._validate_epoch(identity)
            record = self._record_unlocked(identity)
            if record.state is PrefixContextState.POISONED:
                return False
            if record.state is PrefixContextState.RETIRED:
                raise InvalidPrefixContextTransition(
                    "a retired prefix context cannot be poisoned"
                )
            if record.release_started:
                raise InvalidPrefixContextTransition(
                    "resource release has already started for this context"
                )
            record.state = PrefixContextState.POISONED
            record.poison_reason = reason
            self._poisoned_context_ids.add(identity.context_id)
            self._detach_unlocked(record)
            return True

    def current(self, context_id: Hashable) -> PrefixContextIdentity | None:
        hash(context_id)
        with self._lock:
            return self._current.get(context_id)

    def snapshot(self, identity: PrefixContextIdentity) -> PrefixContextSnapshot:
        with self._lock:
            self._validate_epoch(identity)
            return self._snapshot_unlocked(self._record_unlocked(identity))

    def forget_retired(self, identity: PrefixContextIdentity) -> bool:
        """Drop terminal metadata after every external waiter observed it."""

        with self._lock:
            self._validate_epoch(identity)
            record = self._record_unlocked(identity)
            if record.state is not PrefixContextState.RETIRED:
                return False
            self._records.pop(identity)
            return True

    def resource(self, identity: PrefixContextIdentity) -> Any:
        """Return the immutable resource for an ACTIVE/DRAINING version."""

        with self._lock:
            self._validate_epoch(identity)
            record = self._record_unlocked(identity)
            if record.state not in {
                PrefixContextState.ACTIVE,
                PrefixContextState.DRAINING,
            }:
                raise InvalidPrefixContextTransition(
                    f"context in {record.state.value} state has no usable resource"
                )
            return record.resource

    def _new_record_unlocked(
        self,
        context_id: Hashable,
        resource: Any,
        *,
        base_identity: PrefixContextIdentity | None,
    ) -> _ContextRecord:
        version = self._next_version.get(context_id, 0) + 1
        self._next_version[context_id] = version
        identity = PrefixContextIdentity(
            context_id=str(context_id),
            version=version,
            server_epoch=self._server_epoch,
        )
        record = _ContextRecord(
            identity=identity,
            resource=resource,
            base_identity=base_identity,
        )
        self._records[identity] = record
        self._building[context_id] = identity
        return record

    def _record_unlocked(self, identity: PrefixContextIdentity) -> _ContextRecord:
        try:
            return self._records[identity]
        except KeyError as exc:
            raise UnknownPrefixContext(f"unknown prefix context {identity}") from exc

    def _validate_epoch(self, identity: PrefixContextIdentity) -> None:
        if identity.server_epoch != self._server_epoch:
            raise StalePrefixContextVersion(
                "prefix context belongs to another server epoch"
            )

    def _detach_unlocked(self, record: _ContextRecord) -> None:
        identity = record.identity
        if self._current.get(identity.context_id) == identity:
            self._current.pop(identity.context_id, None)
        if self._building.get(identity.context_id) == identity:
            self._building.pop(identity.context_id, None)

    def _prepare_retire_unlocked(self, record: _ContextRecord) -> _ReleaseAction | None:
        if (
            record.state is not PrefixContextState.DRAINING
            or record.retirement_hold
            or record.active_pin is not None
            or record.queue
            or record.release_started
        ):
            return None
        record.release_started = True
        return _ReleaseAction(record.identity, record.resource)

    def _run_release(self, action: _ReleaseAction | None) -> None:
        if action is None:
            return
        try:
            self._release_callback(action.identity, action.resource)
        except Exception as exc:
            with self._lock:
                record = self._record_unlocked(action.identity)
                record.state = PrefixContextState.POISONED
                record.release_error = repr(exc)
                record.poison_reason = "resource release callback failed"
                self._poisoned_context_ids.add(action.identity.context_id)
            raise PrefixContextReleaseError(
                f"failed to release prefix context {action.identity}"
            ) from exc

        with self._lock:
            record = self._record_unlocked(action.identity)
            if record.state is not PrefixContextState.DRAINING:
                raise InvalidPrefixContextTransition(
                    "prefix context changed state during resource release"
                )
            record.resource = None
            record.state = PrefixContextState.RETIRED

    def _snapshot_unlocked(self, record: _ContextRecord) -> PrefixContextSnapshot:
        return PrefixContextSnapshot(
            identity=record.identity,
            state=record.state,
            queued_task_ids=tuple(task.task_id for task in record.queue),
            pinned_task_id=(
                None if record.active_pin is None else record.active_pin.task_id
            ),
            close_reason=record.close_reason,
            poison_reason=record.poison_reason,
            release_error=record.release_error,
        )


__all__ = [
    "DuplicatePrefixTask",
    "InvalidPrefixContextTransition",
    "PrefixContextError",
    "PrefixContextIdentity",
    "PrefixContextRegistry",
    "PrefixContextReleaseError",
    "PrefixContextSnapshot",
    "PrefixContextState",
    "PrefixTask",
    "PrefixTaskPin",
    "StalePrefixContextVersion",
    "UnknownPrefixContext",
]
