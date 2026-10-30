"""Bounded advisory prefix fingerprints for admission, never KV lookup.

A torn or stale snapshot may miss a scheduling opportunity; it cannot authorize
KV reuse or raise the ordinary full-cache admission bound. No request IDs or
prompt text cross this table. Only single-host normal service consumes it.
"""

import hashlib
import time
import numpy as np

from .shared_arr import SharedArray


MAX_PREFIX_HINTS = 64
MIN_PREFIX_HINT_TOKENS = 256
PREFIX_HINT_SHAPE = (MAX_PREFIX_HINTS + 1, 4)


def prefix_hasher(node):
    # A radix node's full prefix stays identical across edge splits and merges.
    # Cache incremental state, avoiding rehashing the whole history per stop.
    pending = []
    while node is not None and not hasattr(node, "_tool_prefix_hasher"):
        pending.append(node)
        node = node.parent
    state = node._tool_prefix_hasher.copy() if node is not None else hashlib.sha256()
    for node in reversed(pending):
        state.update(np.asarray(node.token_id_key, dtype="<i8").tobytes())
        node._tool_prefix_hasher = state.copy()
    return state


def fingerprint(state):
    return tuple(int(value) for value in np.frombuffer(state.digest()[:16], dtype="<i8"))


class ToolCallPrefixHints:
    def __init__(self, rank_in_node, initialize=False):
        self.shared = SharedArray(f"tool_call_prefix_hints_{rank_in_node}", PREFIX_HINT_SHAPE, dtype=np.int64)
        if initialize:
            self.shared.arr[:] = 0

    def publish(self, retention):
        rows = []
        # Earliest expiry first, deterministically bounded even with many small
        # branches. Tiny common scaffolding is not a valuable continuation hint.
        # Missing entries lose only their optional scheduling preference.
        for key, node in retention._frontier.items():
            if node.ref_counter == 0 and node.node_prefix_total_len >= MIN_PREFIX_HINT_TOKENS:
                rows.append((node.node_prefix_total_len, *fingerprint(prefix_hasher(node)), int(key[0] * 1e6)))
                if len(rows) == MAX_PREFIX_HINTS:
                    break
        table = self.shared.arr
        version = int(table[0, 0])
        table[0, 0] = version + 1
        if rows:
            table[1 : len(rows) + 1] = rows
        table[0, 1] = len(rows)
        table[0, 0] = version + 2

    def snapshot(self, now=None):
        table = self.shared.arr
        for _ in range(2):
            before = int(table[0, 0])
            if before % 2:
                continue
            count = max(0, min(MAX_PREFIX_HINTS, int(table[0, 1])))
            rows = table[1 : count + 1].copy()
            if before == int(table[0, 0]):
                return PrefixHintSnapshot(rows, time.monotonic() if now is None else now)
        return PrefixHintSnapshot((), 0)


class PrefixHintSnapshot:
    def __init__(self, rows, now):
        self.by_length = {}
        for length, first, second, deadline in rows:
            if length >= MIN_PREFIX_HINT_TOKENS and deadline > now * 1e6:
                self.by_length.setdefault(int(length), set()).add((int(first), int(second)))

    def matches(self, token_ids, memo=None):
        if not self.by_length:
            return False
        memo = {} if memo is None else memo
        if len(memo) > MAX_PREFIX_HINTS * 2:
            memo.clear()
        lengths = sorted(length for length in self.by_length if length < len(token_ids))
        missing = [length for length in lengths if length not in memo]
        if len(memo) + len(missing) > MAX_PREFIX_HINTS * 2:
            memo.clear()
            missing = lengths
        if missing:
            raw = np.ascontiguousarray(token_ids, dtype="<i8").view(np.uint8)
            state, offset = hashlib.sha256(), 0
            for length in missing:
                state.update(raw[offset * 8 : length * 8])
                memo[length] = fingerprint(state)
                offset = length
        return any(memo[length] in self.by_length[length] for length in lengths)
