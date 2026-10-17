"""Bounded, expiring eviction preferences for tool-result continuations.

The caller supplies a clock shared by all tensor-parallel ranks. Never read a
process-local clock here: different eviction decisions would corrupt TP state.
Leases affect eviction order only; they never acquire KV references.
"""

from sortedcontainers import SortedDict
import numpy as np


class ToolCallCacheRetention:
    def __init__(self, duration, token_budget, discard_node, add_node, fit_prefix=None):
        if duration <= 0 or token_budget < 0:
            raise ValueError("retention duration must be positive and token budget nonnegative")
        self.duration = duration
        self.token_budget = token_budget
        self.clock = 0.0
        self.protected_tokens = 0
        self._discard_node = discard_node
        self._add_node = add_node
        self._fit_prefix = fit_prefix
        self._leases = {}
        self._expiry = SortedDict()
        self._sequence = 0

    def advance_clock(self, now):
        self.clock = max(self.clock, now)
        while self._expiry and self._expiry.peekitem(0)[0][0] <= self.clock:
            self._remove(self._expiry.peekitem(0)[1])

    def protect(self, node):
        if node is None:
            return
        if node.node_prefix_total_len > self.token_budget:
            if self._fit_prefix is not None:
                node = self._fit_prefix(node, self.token_budget)
            else:
                # Hybrid state can only resume at an existing checkpoint.
                while node is not None and node.node_prefix_total_len > self.token_budget:
                    node = node.parent
            if node is None:
                return
        # Protect ancestors as well: evicting a tail must not expose a still
        # useful prefix to immediate eviction. Shared ancestors count only once.
        path = []
        while node.parent is not None:
            path.append(node)
            node = node.parent
        for node in reversed(path):
            self._set(node, self.clock + self.duration)
        while self.protected_tokens > self.token_budget:
            # A protected descendant also retains all its ancestors structurally.
            # Demoting only an ancestor would undercount the effective budget.
            pending = [self._expiry.peekitem(0)[1]]
            while pending:
                victim = pending.pop()
                pending.extend(child for child in victim.children.values() if child.cache_protected)
                self._remove(victim)

    def forget(self, node):
        """Forget a node already removed from the cache's eviction indices."""
        entry = self._leases.pop(node, None)
        if entry is not None:
            key, weight = entry
            del self._expiry[key]
            self.protected_tokens -= weight
        node.cache_protected = False
        node.cache_retention = None

    def complete(self, node):
        """Revoke stale preferences after a continuation answers or hits length.

        Stop at a shared ancestor with another protected branch. Its prefix is
        still useful to that branch's outstanding tool-result continuation.
        """
        while node is not None and node.parent is not None:
            if any(child.cache_protected for child in node.children.values()):
                break
            if node.cache_protected:
                self._remove(node)
            node = node.parent

    def invalidate_checkpoint(self, node, is_resumable):
        """Drop a lost hybrid checkpoint's preference, preserving useful state."""
        while node.parent is not None:
            if any(child.cache_protected for child in node.children.values()):
                return
            if node.cache_protected:
                self._remove(node)
            node = node.parent
            if is_resumable(node):
                return

    def split(self, node, parent):
        """Transfer a compressed radix edge's lease without changing its size."""
        entry = self._leases.get(node)
        if entry is None:
            return
        deadline = entry[0][0]
        # The radix caller has removed node before changing its edge length.
        self.forget(node)
        self._set(parent, deadline)
        self._set(node, deadline)

    def _set(self, node, deadline):
        self._discard_node(node)
        self.forget(node)
        self._sequence += 1
        key = (deadline, self._sequence)
        weight = node.node_value_len
        self._leases[node] = (key, weight)
        self._expiry[key] = node
        self.protected_tokens += weight
        node.cache_protected = True
        node.cache_retention = self
        self._add_node(node)

    def _remove(self, node):
        self._discard_node(node)
        self.forget(node)
        self._add_node(node)


def contains_tool_call(token_ids, markers):
    """Look only at generated tokens, never at tool calls in prompt history.

    This is an advisory marker detector, not the API's tool-call parser. False
    positives can change bounded cache preferences but cannot change responses.
    """
    for marker in markers:
        size = len(marker)
        if size and size <= len(token_ids):
            if isinstance(token_ids, np.ndarray):
                candidates = np.flatnonzero(token_ids[: len(token_ids) - size + 1] == marker[0])
            else:
                candidates = (i for i in range(len(token_ids) - size + 1) if token_ids[i] == marker[0])
            for index in candidates:
                if all(token_ids[index + offset] == token for offset, token in enumerate(marker)):
                    return True
    return False
