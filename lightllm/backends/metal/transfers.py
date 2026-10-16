"""Retain private host snapshots until asynchronous MPS copies complete."""

from collections import deque
import threading

import torch


class HostTransferQueue:
    """Bound host memory and preserve sources reused by request producers."""

    def __init__(self, max_pending=128, max_bytes=64 * 1024 * 1024):
        if max_pending < 1 or max_bytes < 1:
            raise ValueError("Host transfer limits must be positive")
        self.max_pending = max_pending
        self.max_bytes = max_bytes
        self._pending = deque()
        self._bytes = 0
        self._lock = threading.Lock()

    def _reap(self):
        while self._pending and self._pending[0][0].query():
            _, _, size = self._pending.popleft()
            self._bytes -= size

    def move(self, value, memory_format=torch.preserve_format):
        """Return None for unsupported transfers, before allocating or submitting."""
        if not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.layout != torch.strided:
            return None
        size = value.numel() * value.element_size()
        if value.requires_grad or not size or size > self.max_bytes:
            return None

        # A pin-memory manager can reuse the caller's view immediately after
        # move returns. Clone synchronously and retain this independent storage.
        snapshot = value.clone(memory_format=memory_format)
        with self._lock:
            self._reap()
            while self._pending and (len(self._pending) >= self.max_pending or self._bytes + size > self.max_bytes):
                event, source, previous_size = self._pending.popleft()
                # Keep source alive through this blocking wait too.
                event.synchronize()
                self._bytes -= previous_size
                del source
            event = torch.mps.Event(enable_timing=False)
            result = snapshot.to(device="mps", non_blocking=True, memory_format=memory_format)
            try:
                event.record()
            except BaseException:
                # A submitted copy may still reference snapshot on this path.
                torch.mps.synchronize()
                raise
            self._pending.append((event, snapshot, size))
            self._bytes += size
            return result

    def synchronize(self):
        # Serialize with copy submission: clearing after a device wait without
        # this lock could drop a source submitted concurrently after that wait.
        with self._lock:
            torch.mps.synchronize()
            self._pending.clear()
            self._bytes = 0
