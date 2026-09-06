"""DP-local cache-affinity routing based on bounded token-prefix history.

The router owns this heuristic index. It records dispatch history rather than querying
the infer processes' radix trees, so stale entries can only affect placement, not KV
cache correctness.
"""

from __future__ import annotations

import random
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional, Tuple

import xxhash

from lightllm.server.router.batch import Batch, Req
from lightllm.server.router.req_queue.base_queue import BaseQueue

from .base import DpBalancer


PrefixHash = Tuple[int, int]


@dataclass(slots=True)
class DpCacheAwareConfig:
    cache_threshold: float = 0.5
    balance_rel_threshold: float = 1.8
    # DeepSeek-V4 prompt-cache entries are reusable only at 256-token boundaries.
    block_size: int = 256
    max_cache_entries: int = 1_000_000
    evict_entries: int = 10_000


class TokenPrefixCache:
    """Bounded LRU mapping from cumulative token-prefix hashes to local DP indexes."""

    def __init__(self, block_size: int, max_entries: int, evict_entries: int) -> None:
        if block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {block_size}")
        if max_entries < 0:
            raise ValueError(f"max_entries must be >= 0, got {max_entries}")
        if evict_entries < 1:
            raise ValueError(f"evict_entries must be >= 1, got {evict_entries}")
        self.block_size = block_size
        self.max_entries = max_entries
        self.evict_entries = evict_entries
        self._prefix_to_dp: OrderedDict[int, int] = OrderedDict()

    def hash_prefixes(self, prompt_ids) -> List[PrefixHash]:
        cacheable_token_count = max(0, len(prompt_ids) - 1)
        cacheable_token_count = cacheable_token_count // self.block_size * self.block_size
        if cacheable_token_count == 0:
            return []

        token_view = memoryview(prompt_ids)
        item_size = token_view.itemsize
        prompt_bytes = token_view.cast("B")
        # A collision only changes a routing hint; it cannot affect cache correctness.
        # xxh3-64 keeps the 1M-entry index compact and hashes 1M-token prompts faster.
        hasher = xxhash.xxh3_64()
        prefix_hashes = []
        for start in range(0, cacheable_token_count, self.block_size):
            end = start + self.block_size
            hasher.update(prompt_bytes[start * item_size : end * item_size])
            prefix_hashes.append((hasher.intdigest(), end))
        return prefix_hashes

    def match(self, prefix_hashes: List[PrefixHash]) -> Tuple[Optional[int], int]:
        for prefix_hash, token_count in reversed(prefix_hashes):
            try:
                dp_index = self._prefix_to_dp[prefix_hash]
            except KeyError:
                continue
            self._prefix_to_dp.move_to_end(prefix_hash)
            return dp_index, token_count
        return None, 0

    def insert(self, prefix_hashes: List[PrefixHash], dp_index: int, start_index: int = 0) -> None:
        for prefix_index in range(start_index, len(prefix_hashes)):
            prefix_hash = prefix_hashes[prefix_index][0]
            self._prefix_to_dp[prefix_hash] = dp_index
            self._prefix_to_dp.move_to_end(prefix_hash)

        if len(self._prefix_to_dp) > self.max_entries:
            evict_count = len(self._prefix_to_dp) - self.max_entries + self.evict_entries
            for _ in range(min(evict_count, len(self._prefix_to_dp))):
                self._prefix_to_dp.popitem(last=False)

    def __len__(self) -> int:
        return len(self._prefix_to_dp)


class DpCacheAwareBalancer(DpBalancer):
    """Route matching token prefixes to the same local DP unless load requires rebalancing."""

    def __init__(
        self,
        dp_size_in_node: int,
        inner_queues: List[BaseQueue],
        config: Optional[DpCacheAwareConfig] = None,
    ) -> None:
        super().__init__(dp_size_in_node, inner_queues)
        self.config = config or DpCacheAwareConfig()
        self.prefix_cache = TokenPrefixCache(
            block_size=self.config.block_size,
            max_entries=self.config.max_cache_entries,
            evict_entries=self.config.evict_entries,
        )

    def assign_reqs_to_dp(self, current_batch: Batch, reqs_waiting_for_dp_index: List[List[Req]]) -> None:
        if not reqs_waiting_for_dp_index:
            return

        current_load_per_dp = [0 for _ in range(self.dp_size_in_node)]
        if current_batch is not None:
            current_load_per_dp = current_batch.get_all_dp_req_num()
        total_load_per_dp = [
            current_load_per_dp[dp_index] + len(self.inner_queues[dp_index].waiting_req_list)
            for dp_index in range(self.dp_size_in_node)
        ]

        for req_group in reqs_waiting_for_dp_index:
            first_req = req_group[0]
            linked_prompt_ids = False
            if not hasattr(first_req, "shm_prompt_ids"):
                first_req.link_prompt_ids_shm_array()
                linked_prompt_ids = True
            try:
                prefix_hashes = self.prefix_cache.hash_prefixes(first_req.get_prompt_ids_numpy())
            finally:
                if linked_prompt_ids:
                    first_req.shm_prompt_ids.detach_shm()
                    del first_req.shm_prompt_ids

            cache_dp_index = None
            matched_token_count = 0
            if not first_req.sample_params.disable_prompt_cache:
                matched_dp_index, matched_token_count = self.prefix_cache.match(prefix_hashes)
                match_rate = matched_token_count / first_req.input_len if first_req.input_len else 0.0
                if match_rate > self.config.cache_threshold:
                    cache_dp_index = matched_dp_index

            idle_dp_indexes = [dp_index for dp_index, load in enumerate(total_load_per_dp) if load == 0]
            if idle_dp_indexes:
                if cache_dp_index in idle_dp_indexes:
                    selected_dp_index = cache_dp_index
                else:
                    selected_dp_index = random.choice(idle_dp_indexes)
            else:
                min_load = min(total_load_per_dp)
                least_loaded_dp_indexes = [
                    dp_index for dp_index, load in enumerate(total_load_per_dp) if load == min_load
                ]
                least_loaded_dp_index = random.choice(least_loaded_dp_indexes)
                if cache_dp_index is None:
                    selected_dp_index = least_loaded_dp_index
                else:
                    group_load = len(req_group)
                    cache_projected_load = total_load_per_dp[cache_dp_index] + group_load
                    least_projected_load = total_load_per_dp[least_loaded_dp_index] + group_load
                    if cache_projected_load > least_projected_load * self.config.balance_rel_threshold:
                        selected_dp_index = least_loaded_dp_index
                    else:
                        selected_dp_index = cache_dp_index

            for req in req_group:
                req.sample_params.suggested_dp_index = selected_dp_index
            self.inner_queues[selected_dp_index].extend(req_group)
            total_load_per_dp[selected_dp_index] += len(req_group)
            insert_start_index = 0
            if cache_dp_index == selected_dp_index:
                insert_start_index = (matched_token_count + self.config.block_size - 1) // self.config.block_size
            self.prefix_cache.insert(prefix_hashes, selected_dp_index, start_index=insert_start_index)

        reqs_waiting_for_dp_index.clear()
