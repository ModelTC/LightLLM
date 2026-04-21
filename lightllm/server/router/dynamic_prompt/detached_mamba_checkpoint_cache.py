from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import xxhash
from sortedcontainers import SortedSet

from lightllm.server.router.dynamic_prompt.radix_cache import time_gen


@dataclass(eq=False)
class DetachedMambaCheckpoint:
    key: Tuple[int, int]
    buffer_idx: int
    prefix_len: int
    time_id: int = field(default_factory=time_gen.generate_time_id)

    def touch(self):
        self.time_id = time_gen.generate_time_id()


class DetachedMambaCheckpointCache:
    """Keep page-aligned mamba checkpoints alive after their KV node is evicted.

    The checkpoint storage reuses the existing mamba buffer pool; this cache only
    decouples checkpoint lifetime from radix-tree node lifetime when CPU KV cache
    can later restore the matching KV pages.
    """

    def __init__(self, buffer_mem_manager, token_page_size: int):
        self.buffer_mem_manager = buffer_mem_manager
        self.token_page_size = token_page_size
        self._checkpoints: Dict[Tuple[int, int], DetachedMambaCheckpoint] = {}
        self._evict_set = SortedSet(key=lambda x: x.time_id)

    def _to_token_list(self, tokens: Iterable[int]) -> list[int]:
        if isinstance(tokens, torch.Tensor):
            return tokens.cpu().tolist()
        if isinstance(tokens, np.ndarray):
            return tokens.tolist()
        return list(tokens)

    def _build_key(self, tokens: Iterable[int]) -> Optional[Tuple[Tuple[int, int], int]]:
        token_list = self._to_token_list(tokens)
        prefix_len = len(token_list)
        if prefix_len == 0 or prefix_len % self.token_page_size != 0:
            return None

        hsum = xxhash.xxh3_128()
        for page_idx in range(prefix_len // self.token_page_size):
            start = page_idx * self.token_page_size
            end = start + self.token_page_size
            chunk_np = np.array(token_list[start:end], dtype=np.uint64)
            hsum.update(chunk_np.tobytes())

        return (prefix_len // self.token_page_size, hsum.intdigest()), prefix_len

    def add_checkpoint(self, tokens: Iterable[int], buffer_idx: int) -> bool:
        built = self._build_key(tokens)
        if built is None:
            return False

        key, prefix_len = built
        old_checkpoint = self._checkpoints.pop(key, None)
        if old_checkpoint is not None:
            self._evict_set.discard(old_checkpoint)
            self.buffer_mem_manager.free([old_checkpoint.buffer_idx])

        checkpoint = DetachedMambaCheckpoint(key=key, buffer_idx=buffer_idx, prefix_len=prefix_len)
        self._checkpoints[key] = checkpoint
        self._evict_set.add(checkpoint)
        return True

    def match_prompt_prefix(
        self, prompt_tokens: Iterable[int], max_prefix_len: int
    ) -> Optional[DetachedMambaCheckpoint]:
        page_num_limit = max_prefix_len // self.token_page_size
        if page_num_limit <= 0:
            return None

        token_list = self._to_token_list(prompt_tokens)
        if len(token_list) < page_num_limit * self.token_page_size:
            page_num_limit = len(token_list) // self.token_page_size
        if page_num_limit <= 0:
            return None

        hsum = xxhash.xxh3_128()
        best_checkpoint = None
        for page_idx in range(page_num_limit):
            start = page_idx * self.token_page_size
            end = start + self.token_page_size
            chunk_np = np.array(token_list[start:end], dtype=np.uint64)
            hsum.update(chunk_np.tobytes())
            key = (page_idx + 1, hsum.intdigest())
            checkpoint = self._checkpoints.get(key)
            if checkpoint is not None:
                best_checkpoint = checkpoint

        if best_checkpoint is None:
            return None

        self._evict_set.discard(best_checkpoint)
        best_checkpoint.touch()
        self._evict_set.add(best_checkpoint)
        return best_checkpoint

    def evict_to_get_enough_buffer(self, need_buffer_num: int):
        if need_buffer_num <= 0:
            return

        release_buffers = []
        while need_buffer_num > 0 and self._evict_set:
            checkpoint = self._evict_set.pop(0)
            self._checkpoints.pop(checkpoint.key, None)
            release_buffers.append(checkpoint.buffer_idx)
            need_buffer_num -= 1

        if release_buffers:
            self.buffer_mem_manager.free(release_buffers)
