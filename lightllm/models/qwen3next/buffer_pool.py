# lightllm/models/qwen3next/buffer_pool.py
import torch
from typing import Dict, Tuple


class Qwen3NextBufferPool:
    """
    Buffer pool for Qwen3Next inference to reduce allocations.

    NOT thread-safe. Each GPU worker process should have its own pool instance.

    Manages reusable buffers for:
    - Attention norm outputs
    - FFN norm outputs
    - FFN intermediate activations
    - GDN intermediate tensors
    """

    def __init__(self, enable_stats: bool = False, max_buffers: int = 64):
        self._buffers: Dict[Tuple[tuple, torch.dtype, torch.device], torch.Tensor] = {}
        self._in_use: set = set()
        self._max_buffers = max_buffers
        self._access_order: list = []  # Track LRU order
        self._enable_stats = enable_stats
        self._stats = {"hits": 0, "misses": 0, "peak_buffers": 0, "evictions": 0} if enable_stats else None

    def get_buffer(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Get a buffer from the pool or allocate a new one."""
        key = (shape, dtype, device)

        # Check if we have a matching buffer not in use
        if key in self._buffers and key not in self._in_use:
            self._in_use.add(key)
            # Update LRU order
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            if self._enable_stats:
                self._stats["hits"] += 1
            return self._buffers[key]

        # Evict oldest unused buffer if at capacity
        if len(self._buffers) >= self._max_buffers:
            self._evict_one()

        # Allocate new buffer
        buffer = torch.empty(shape, dtype=dtype, device=device)
        self._buffers[key] = buffer
        self._in_use.add(key)
        self._access_order.append(key)
        if self._enable_stats:
            self._stats["misses"] += 1
            self._stats["peak_buffers"] = max(self._stats["peak_buffers"], len(self._buffers))
        return buffer

    def _evict_one(self):
        """Evict oldest unused buffer (LRU)."""
        for key in self._access_order:
            if key not in self._in_use and key in self._buffers:
                del self._buffers[key]
                self._access_order.remove(key)
                if self._enable_stats:
                    self._stats["evictions"] += 1
                return

    def release_all(self):
        """Release all buffers back to the pool (call after forward pass)."""
        self._in_use.clear()

    def clear(self):
        """Clear all buffers (call when changing batch size significantly)."""
        self._buffers.clear()
        self._in_use.clear()
        self._access_order.clear()

    def get_stats(self):
        """Return buffer pool statistics (if enabled)."""
        return self._stats.copy() if self._stats else None
