"""CPU-only cumulative engine counters and monotonic reporting windows."""

import time
import atomics
import numpy as np

from lightllm.server.core.objs.shm_array import ShmArray


class EngineCounters:
    """One writer per local DP row (TP leader); the router is the sole reader.

    Workers attach after the router initializes the array. Never reset a live row:
    sampling cumulative totals keeps short requests and overlapping windows exact.
    """

    def __init__(self, dp_size, dp_rank=None, *, owner=False, mtp_enabled=False):
        self.storage = ShmArray("engine_token_counters", (dp_size, 6), np.int64)
        if owner:
            self.storage.create_shm()
            self.storage.arr.fill(0)
        else:
            self.storage.link_shm()
        self.dp_rank = dp_rank
        self.mtp_enabled = mtp_enabled
        self._speculation_snapshot = np.zeros((dp_size, 3), dtype=np.int64)

    def add(self, prefill_tokens=0, output_tokens=0):
        row = self.storage.arr[self.dp_rank]
        row[0] += prefill_tokens
        row[1] += output_tokens

    def totals(self):
        return tuple(int(value) for value in self.storage.arr[:, :2].sum(axis=0))

    def add_speculation(self, accepted, verified, rounds):
        row = self.storage.arr[self.dp_rank]
        with atomics.atomicview(row[5:].view(np.uint8), atype=atomics.UINT) as version:
            version.inc()
            row[2:5] += (accepted, verified, rounds)
            version.inc()

    def speculation_totals(self):
        """Keep the previous complete row if a writer overlaps the snapshot."""
        for rank, row in enumerate(self.storage.arr):
            with atomics.atomicview(row[5:].view(np.uint8), atype=atomics.UINT) as version:
                before = version.load()
                if before % 2:
                    continue
                values = row[2:5].copy()
                if before == version.fetch_add(0):
                    self._speculation_snapshot[rank] = values
        return tuple(int(value) for value in self._speculation_snapshot.sum(axis=0))


class EngineThroughput:
    """Window rates include running requests; request success is counted at release."""

    def __init__(self, counters):
        self.counters = counters
        self.completed = 0
        self.last_prefill, self.last_output = counters.totals()
        self.last_speculation = counters.speculation_totals() if counters.mtp_enabled else (0, 0, 0)
        self.last_completed = 0
        self.last_time = time.monotonic()

    def finish(self, req):
        if req.is_stop_matched or (
            not req.is_aborted and (req.finish_status.is_stopped() or req.finish_status.is_finished_length())
        ):
            self.completed += 1

    def has_activity(self):
        prefill, output = self.counters.totals()
        return (prefill, output, self.completed) != (
            self.last_prefill,
            self.last_output,
            self.last_completed,
        ) or (self.counters.mtp_enabled and self.counters.speculation_totals() != self.last_speculation)

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
        if self.counters.mtp_enabled:
            speculation = self.counters.speculation_totals()
            accepted, verified, rounds = (cur - prev for cur, prev in zip(speculation, self.last_speculation))
            result.update(
                mtp_accepted_tokens=accepted,
                mtp_verified_draft_tokens=verified,
                mtp_rounds=rounds,
                mtp_acceptance_pct=round(100 * accepted / verified, 2) if verified else None,
                mtp_mean_acceptance_length=round(1 + accepted / rounds, 2) if rounds else None,
                mtp_accepted_tok_s=round(accepted / elapsed, 2) if elapsed > 0 else 0.0,
                mtp_verified_draft_tok_s=round(verified / elapsed, 2) if elapsed > 0 else 0.0,
            )
            self.last_speculation = speculation
        self.last_prefill, self.last_output, self.last_completed = prefill, output, self.completed
        self.last_time = now
        return result
