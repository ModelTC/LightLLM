"""Per-session RNG state cache for the x2i server.

In interleaved image-text scenarios, only the very first image of a chat
session needs an explicit seed; subsequent images should continue from the
RNG state left by the previous image generation. With multiple chat sessions
running concurrently, the global torch / cuda / numpy / random RNG state is
shared across sessions, so seeding for session B's first image would corrupt
session A's continuation.

This cache snapshots the RNG state after each generation per session_id and
restores it before the next generation of the same session, so that each
session sees a private, deterministic RNG stream.
"""

import random
import time
from typing import Dict, Tuple

import numpy as np
import torch

from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


def capture_rng_state() -> Dict:
    state = {
        "random": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict) -> None:
    random.setstate(state["random"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


class RngStateCache:
    """LRU + TTL cache for RNG snapshots keyed by chat session id.

    - `save(session_id)`: snapshot current global RNG state for the session.
    - `restore(session_id)`: restore previously saved state, returns True on hit.
    - `discard(session_id)`: drop the session entry if any.
    - Entries expire after `ttl_seconds` of inactivity, and the cache is hard
      capped at `max_size` (oldest evicted first) to bound memory.
    """

    def __init__(self, max_size: int = 1024, ttl_seconds: float = 600.0):
        self._max_size = max_size
        self._ttl = ttl_seconds
        # session_id -> (last_used_ts, state_dict)
        self._cache: Dict[int, Tuple[float, Dict]] = {}

    def save(self, session_id: int) -> None:
        if session_id == 0:
            return
        self._cache[session_id] = (time.time(), capture_rng_state())
        self._evict()

    def restore(self, session_id: int) -> bool:
        if session_id == 0:
            return False
        item = self._cache.get(session_id)
        if item is None:
            return False
        ts, state = item
        if self._ttl > 0 and time.time() - ts > self._ttl:
            self._cache.pop(session_id, None)
            logger.warning(f"RNG state for session {session_id} expired (TTL={self._ttl}s)")
            return False
        restore_rng_state(state)
        self._cache[session_id] = (time.time(), state)
        return True

    def discard(self, session_id: int) -> None:
        self._cache.pop(session_id, None)

    def _evict(self) -> None:
        if self._ttl > 0:
            now = time.time()
            expired = [sid for sid, (ts, _) in self._cache.items() if now - ts > self._ttl]
            for sid in expired:
                self._cache.pop(sid, None)
        if len(self._cache) > self._max_size:
            sorted_items = sorted(self._cache.items(), key=lambda kv: kv[1][0])
            for sid, _ in sorted_items[: len(self._cache) - self._max_size]:
                self._cache.pop(sid, None)
