from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .objs import ActionTaskIdentity


class TaskSubmissionDecision(str, Enum):
    ACCEPT = "accept"
    DUPLICATE = "duplicate"
    STALE = "stale"


@dataclass
class _TaskRecord:
    identity: ActionTaskIdentity
    terminal: bool = False


class ActionTaskRegistry:
    """Process-local exactly-once and ABA guard for action task delivery.

    Shared-memory identity checks remain authoritative for publishing results;
    this registry prevents the manager from executing a duplicated task twice.
    A slot may accept another task only when its generation increases.
    """

    def __init__(self, max_history: int = 4096):
        if max_history <= 0:
            raise ValueError("max_history must be positive")
        self.max_history = max_history
        self._latest_by_slot: dict[int, ActionTaskIdentity] = {}
        self._records: dict[ActionTaskIdentity, _TaskRecord] = {}
        self._terminal_order: list[ActionTaskIdentity] = []

    def submit(self, identity: ActionTaskIdentity) -> TaskSubmissionDecision:
        if identity in self._records:
            return TaskSubmissionDecision.DUPLICATE

        latest = self._latest_by_slot.get(identity.slot_index)
        if latest is not None and identity.generation <= latest.generation:
            return TaskSubmissionDecision.STALE

        self._latest_by_slot[identity.slot_index] = identity
        self._records[identity] = _TaskRecord(identity=identity)
        return TaskSubmissionDecision.ACCEPT

    def mark_terminal(self, identity: ActionTaskIdentity) -> bool:
        record = self._records.get(identity)
        if record is None or record.terminal:
            return False
        record.terminal = True
        self._terminal_order.append(identity)
        self._trim_history()
        return True

    def is_current(self, identity: ActionTaskIdentity) -> bool:
        return self._latest_by_slot.get(identity.slot_index) == identity

    def contains(self, identity: ActionTaskIdentity) -> bool:
        return identity in self._records

    def _trim_history(self):
        while len(self._terminal_order) > self.max_history:
            identity = self._terminal_order.pop(0)
            if self._latest_by_slot.get(identity.slot_index) == identity:
                # The newest identity for a slot is retained until a newer
                # generation proves that this slot has been recycled.
                self._terminal_order.append(identity)
                break
            self._records.pop(identity, None)
