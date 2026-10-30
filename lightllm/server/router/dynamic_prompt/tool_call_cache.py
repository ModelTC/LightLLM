"""Bounded, expiring eviction preferences for tool-result continuations.

The caller supplies a clock shared by all tensor-parallel ranks. Never read a
process-local clock here: different eviction decisions would corrupt TP state.
Leases change eviction preference and publish idle coverage for normal-service
admission. They never acquire KV references.
"""

from sortedcontainers import SortedDict
import numpy as np


class ToolCallCacheRetention:
    def __init__(self, duration, token_budget, discard_node, add_node, fit_prefix=None, publish_reserved=None):
        if duration <= 0 or token_budget < 0:
            raise ValueError("retention duration must be positive and token budget nonnegative")
        self.duration = duration
        self.token_budget = token_budget
        self.clock = 0.0
        self.protected_tokens = 0
        self.reserved_tokens = 0
        self._publish_reserved = publish_reserved
        self._discard_node = discard_node
        self._add_node = add_node
        self._fit_prefix = fit_prefix
        self._leases = {}
        self._expiry = SortedDict()
        # Only protected endpoints compete for budget. An ancestor is shared
        # storage, not an independently evictable continuation.
        self._frontier = SortedDict()
        self._sequence = 0
        self._stats = dict.fromkeys(
            (
                "tool_stops",
                "consumed_tokens",
                "budget_tokens",
                "expired_tokens",
                "evicted_tokens",
                "completed_tokens",
                "checkpoint_tokens",
            ),
            0,
        )

    def get_stats(self):
        return {
            **self._stats,
            "protected_tokens": self.protected_tokens,
            "reserved_tokens": self.reserved_tokens,
            "waiting_endpoints": len(self._frontier),
        }

    def _reserve(self, delta):
        self.reserved_tokens += delta
        assert 0 <= self.reserved_tokens <= self.protected_tokens
        if self._publish_reserved is not None:
            self._publish_reserved(self.reserved_tokens)

    def reference_changed(self, node, referenced):
        """Only idle protected KV reserves admission space; active refs own it."""
        if node.cache_protected:
            self._reserve(-node.node_value_len if referenced else node.node_value_len)

    def advance_clock(self, now):
        self.clock = max(self.clock, now)
        while self._expiry and self._expiry.peekitem(0)[0][0] <= self.clock:
            self._remove(self._expiry.peekitem(0)[1], "expired_tokens")

    def protect(self, node, reusable_prefix_len=None):
        self._stats["tool_stops"] += 1
        if node is None:
            return
        limit = self.token_budget if reusable_prefix_len is None else min(self.token_budget, reusable_prefix_len)
        if limit <= 0:
            return
        if node.node_prefix_total_len > limit:
            if self._fit_prefix is not None:
                node = self._fit_prefix(node, limit)
            else:
                # Hybrid state can only resume at an existing checkpoint.
                while node is not None and node.node_prefix_total_len > limit:
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
        self._trim_budget()

    def _refresh_frontier(self, node):
        entry = self._leases.get(node)
        if entry is None:
            return
        key = entry[0]
        if any(child.cache_protected for child in node.children.values()):
            self._frontier.pop(key, None)
        else:
            self._frontier[key] = node

    def _trim_budget(self):
        while self.protected_tokens > self.token_budget:
            victim = self._frontier.peekitem(0)[1]
            # Keep as much of the older continuation as the remaining budget
            # permits. Full attention can split an edge at a page boundary;
            # hybrid attention must fall back to a real recurrent checkpoint.
            limit = max(0, victim.node_prefix_total_len - (self.protected_tokens - self.token_budget))
            if self._fit_prefix is not None:
                keep = self._fit_prefix(victim, limit)
            else:
                keep = victim
                while keep.parent is not None and keep.node_prefix_total_len > limit:
                    keep = keep.parent
            while victim is not keep and victim.parent is not None:
                if any(child.cache_protected for child in victim.children.values()):
                    break
                if victim.cache_protected:
                    self._remove(victim, "budget_tokens")
                victim = victim.parent

    def consume(self, node):
        """A matched continuation is now pinned by the request, not waiting.

        Release its lease budget immediately. Other waiting descendants still
        own their shared ancestors. A further tool stop will acquire a fresh
        lease for the updated prefix when this request finishes.
        """
        if node is not None:
            assert node.ref_counter > 0
            self.complete(node, "consumed_tokens")

    def forget(self, node, evicted=False):
        """Forget a node already removed from the cache's eviction indices."""
        entry = self._leases.pop(node, None)
        if entry is not None:
            key, weight = entry
            del self._expiry[key]
            self._frontier.pop(key, None)
            # Remove the idle reservation before reducing the total coverage.
            if node.ref_counter == 0:
                self._reserve(-weight)
            self.protected_tokens -= weight
            if evicted:
                self._stats["evicted_tokens"] += weight
        node.cache_protected = False
        node.cache_retention = None
        self._refresh_frontier(node.parent)

    def complete(self, node, reason="completed_tokens"):
        """Revoke stale preferences after a continuation answers or hits length.

        Stop at a shared ancestor with another protected branch. Its prefix is
        still useful to that branch's outstanding tool-result continuation.
        """
        while node is not None and node.parent is not None:
            if any(child.cache_protected for child in node.children.values()):
                break
            if node.cache_protected:
                self._remove(node, reason)
            node = node.parent

    def invalidate_checkpoint(self, node, is_resumable):
        """Drop a lost hybrid checkpoint's preference, preserving useful state."""
        while node.parent is not None:
            if any(child.cache_protected for child in node.children.values()):
                return
            if node.cache_protected:
                self._remove(node, "checkpoint_tokens")
            node = node.parent
            if is_resumable(node):
                return

    def split(self, node, parent):
        """Transfer a compressed radix edge's lease without changing its size."""
        entry = self._leases.get(node)
        if entry is None:
            return
        deadline, priority = entry[0][:2]
        # The radix caller has removed node before changing its edge length.
        self.forget(node)
        self._set(parent, deadline, priority)
        self._set(node, deadline, priority)

    def _set(self, node, deadline, priority=None):
        entry = self._leases.get(node)
        if entry is not None and entry[1] == node.node_value_len:
            # Renewal changes only expiry, not eviction priority or idle bytes.
            # Avoid toggling shared reservation counters and all eviction sets
            # along the entire history of a long-running agent.
            old_key, weight = entry
            del self._expiry[old_key]
            self._frontier.pop(old_key, None)
            self._sequence += 1
            key = (deadline, self._sequence if priority is None else priority, self._sequence)
            self._leases[node] = (key, weight)
            self._expiry[key] = node
            self._refresh_frontier(node)
            return
        self._discard_node(node)
        self.forget(node)
        self._sequence += 1
        key = (deadline, self._sequence if priority is None else priority, self._sequence)
        weight = node.node_value_len
        self._leases[node] = (key, weight)
        self._expiry[key] = node
        self.protected_tokens += weight
        if node.ref_counter == 0:
            self._reserve(weight)
        node.cache_protected = True
        node.cache_retention = self
        self._add_node(node)
        self._refresh_frontier(node)
        self._refresh_frontier(node.parent)

    def _remove(self, node, reason):
        self._stats[reason] += self._leases[node][1]
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
