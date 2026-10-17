"""CPU-only cumulative engine counters and monotonic reporting windows."""

import time
import numpy as np

from lightllm.server.core.objs.shm_array import ShmArray


class EngineCounters:
    """One writer per local DP row (TP leader); the router is the sole reader.

    Workers attach after the router initializes the array. Never reset a live row:
    sampling cumulative totals keeps short requests and overlapping windows exact.
    """

    def __init__(self, dp_size, dp_rank=None, *, owner=False):
        self.storage = ShmArray("engine_token_counters", (dp_size, 2), np.int64)
        if owner:
            self.storage.create_shm()
            self.storage.arr.fill(0)
        else:
            self.storage.link_shm()
        self.dp_rank = dp_rank

    def add(self, prefill_tokens=0, output_tokens=0):
        row = self.storage.arr[self.dp_rank]
        row[0] += prefill_tokens
        row[1] += output_tokens

    def totals(self):
        return tuple(int(value) for value in self.storage.arr.sum(axis=0))


class EngineThroughput:
    """Window rates include running requests; request success is counted at release."""

    def __init__(self, counters):
        self.counters = counters
        self.completed = 0
        self.last_prefill, self.last_output = counters.totals()
        self.last_completed = 0
        self.last_time = time.monotonic()

    def finish(self, req):
        if not req.is_aborted and (req.finish_status.is_stopped() or req.finish_status.is_finished_length()):
            self.completed += 1

    def has_activity(self):
        prefill, output = self.counters.totals()
        return (prefill, output, self.completed) != (self.last_prefill, self.last_output, self.last_completed)

    def resume(self, since=None):
        """Start a new busy window without losing tokens from a short burst."""
        self.last_time = time.monotonic() if since is None else since

    def snapshot(self):
        now = time.monotonic()
        elapsed = now - self.last_time
        prefill, output = self.counters.totals()
        counts = (prefill - self.last_prefill, output - self.last_output, self.completed - self.last_completed)
        result = dict(zip(("prefill_tokens", "output_tokens", "completed_reqs"), counts))
        result.update(
            window_s=round(elapsed, 6),
            prefill_tok_s=round(counts[0] / elapsed, 2) if elapsed > 0 else 0.0,
            output_tok_s=round(counts[1] / elapsed, 2) if elapsed > 0 else 0.0,
            completed_req_s=round(counts[2] / elapsed, 2) if elapsed > 0 else 0.0,
        )
        self.last_prefill, self.last_output, self.last_completed = prefill, output, self.completed
        self.last_time = now
        return result
