from typing import Set, Protocol, List, Optional, Tuple

import torch
from sortedcontainers import SortedSet

from lightllm.server.router.dynamic_prompt.radix_cache import RadixCache, TreeNode
from lightllm.common.mamba_cache_mem_manager.cache_manager import MambaCacheManager
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class HybridRadixCache(RadixCache):
    def __init__(self, unique_name, total_token_num, rank_in_node, kv_cache_mem_manager):
        super().__init__(unique_name, total_token_num, rank_in_node, kv_cache_mem_manager)
        assert hasattr(kv_cache_mem_manager, "mamba_cache_mem_manager")
        self.buffer_mem_manager: MambaCacheManager = kv_cache_mem_manager.mamba_cache_mem_manager
        self.evict_buffer_set: Set[TreeNode] = SortedSet(key=lambda x: (x.is_hotspot, x.buffer_time))

        # Adaptive threshold state
        self.min_insert_threshold = 1024
        self.MIN_THRESHOLD = 256
        self.MAX_THRESHOLD = 16384
        self.adjust_interval = 100
        self.buffer_insert_count = 0
        self.buffer_hit_count = 0
        self.buffer_waste_count = 0

    def match_prefix(self, key, update_refs=False):
        assert len(key) != 0
        ans_value_list = []
        tree_node = self._match_prefix_helper(self.root_node, key, ans_value_list, update_refs=update_refs)
        miss_prefix_len = 0
        while tree_node != self.root_node and tree_node.buffer_idx is None:
            miss_prefix_len += len(ans_value_list[-1]) if ans_value_list else 0

            next_node = tree_node.parent

            if update_refs:
                # Undo the ref increment from _match_prefix_helper.
                # Do NOT destroy nodes here — that caused a cascade where each
                # destroyed child turned its parent into a leaf, which was then
                # also destroyed, silently wiping the entire prefix chain.
                # Unreferenced leaves will be reclaimed by the normal eviction path.
                if tree_node.is_leaf():
                    self.evict_tree_set.discard(tree_node)
                if tree_node.ref_counter == 1:
                    self.refed_tokens_num.arr[0] -= len(tree_node.token_mem_index_value)
                tree_node.ref_counter -= 1
                if tree_node.is_leaf():
                    self.evict_tree_set.add(tree_node)

            ans_value_list.pop()
            tree_node = next_node

        if tree_node == self.root_node:
            return None, miss_prefix_len, None

        # Mark buffer node as hit when update_refs is True
        if update_refs:
            tree_node.was_hit = True

        update_node = tree_node
        while update_node != self.root_node:
            if update_node.buffer_idx is not None:
                self.evict_buffer_set.discard(update_node)
                update_node.update_buffer_time()
                self.evict_buffer_set.add(update_node)
            update_node = update_node.parent

        value = torch.concat(ans_value_list)
        return tree_node, miss_prefix_len, value

    def add_buffer_idx_to_node(self, node: TreeNode, buffer_idx: int, is_hotspot: bool = False):
        """Set buffer_idx for a node and add it to evict_buffer_set."""
        self.evict_buffer_set.discard(node)
        if node.is_leaf():
            self.evict_tree_set.discard(node)
        if node.buffer_idx is not None:
            self.buffer_mem_manager.free([node.buffer_idx])
        node.buffer_idx = buffer_idx
        node.is_hotspot = is_hotspot
        node.was_hit = False
        node.update_buffer_time()
        self.evict_buffer_set.add(node)
        if node.is_leaf():
            self.evict_tree_set.add(node)
        if is_hotspot:
            self.buffer_insert_count += 1
        return

    def free_radix_cache_to_get_enough_buffer(self, need_buffer_num):
        if need_buffer_num > self.buffer_mem_manager.can_use_mem_size:
            need_evict_buffer_num = need_buffer_num - self.buffer_mem_manager.can_use_mem_size
            release_buffers = []
            release_kv_mems = []

            def release_buffer(buffer_idx):
                release_buffers.append(buffer_idx)
                return

            def release_kv_mem(token_mem_index_value):
                release_kv_mems.append(token_mem_index_value)
                return

            self._evict_buffer(need_evict_buffer_num, release_buffer, release_kv_mem)
            if len(release_buffers) > 0:
                self.buffer_mem_manager.free(release_buffers)
            if len(release_kv_mems) > 0:
                kv_mem_index = torch.concat(release_kv_mems)
                self.mem_manager.free(kv_mem_index)
        return

    def _evict_buffer(self, need_evict_buffer_num, evict_buffer_callback, evict_token_callback):
        while need_evict_buffer_num > 0:
            node = self.evict_buffer_set.pop(0)
            assert node.buffer_idx is not None

            # Track waste/hit
            if node.was_hit:
                self.buffer_hit_count += 1
            else:
                self.buffer_waste_count += 1

            evict_buffer_callback(node.buffer_idx)
            node.buffer_idx = None
            node.is_hotspot = False
            node.was_hit = False
            need_evict_buffer_num -= 1

            # R4 cascading cleanup: if node is now an unreferenced leaf, destroy it
            if node.is_leaf() and node.ref_counter == 0:
                self.evict_tree_set.discard(node)
                evict_token_callback(node.token_mem_index_value)
                self.tree_total_tokens_num.arr[0] -= len(node.token_mem_index_value)
                parent_node = node.parent
                parent_node.remove_child(node)
                if parent_node.is_leaf():
                    self.evict_tree_set.add(parent_node)

        self._maybe_adjust_threshold()
        return

    def free_radix_cache_to_get_enough_token(self, need_token_num):
        assert self.mem_manager is not None
        if need_token_num > self.mem_manager.can_use_mem_size:
            need_evict_token_num = need_token_num - self.mem_manager.can_use_mem_size
            release_mems = []

            def release_mem(mem_index):
                release_mems.append(mem_index)
                return

            release_buffers = []

            def release_buffer(buffer_idx):
                release_buffers.append(buffer_idx)
                return

            self.evict(need_evict_token_num, release_buffer, release_mem)
            mem_index = torch.concat(release_mems)
            self.mem_manager.free(mem_index)
            if len(release_buffers) > 0:
                self.buffer_mem_manager.free(release_buffers)
        return

    def evict(self, need_remove_tokens, evict_buffer_callback, evict_callback):
        if self.tree_total_tokens_num.arr[0] - self.refed_tokens_num.arr[0] < need_remove_tokens:
            assert False, f"""can not free tree tokens {need_remove_tokens},
                              tree_total_tokens_num {self.tree_total_tokens_num.arr[0]},
                              refed_tokens_num {self.refed_tokens_num.arr[0]}"""
        num_evicted = 0
        while num_evicted < need_remove_tokens:
            node: TreeNode = self.evict_tree_set.pop(0)
            assert (
                node.ref_counter == 0 and len(node.children) == 0 and node != self.root_node
            ), f"error evict tree node state: {node.ref_counter}, {len(node.children)}"
            num_evicted += len(node.token_mem_index_value)
            evict_callback(node.token_mem_index_value)
            if node.buffer_idx is not None:
                self.evict_buffer_set.discard(node)
                evict_buffer_callback(node.buffer_idx)
                node.buffer_idx = None
            # update total token num
            self.tree_total_tokens_num.arr[0] -= len(node.token_mem_index_value)
            parent_node: TreeNode = node.parent
            parent_node.remove_child(node)
            if parent_node.is_leaf():
                self.evict_tree_set.add(parent_node)

        return

    def available_opportunistic_buffer_count(self) -> int:
        """Count how many buffers can be obtained without evicting hotspot buffers."""
        free = self.buffer_mem_manager.can_use_mem_size
        non_hotspot = sum(1 for n in self.evict_buffer_set if not n.is_hotspot)
        return free + non_hotspot

    def _maybe_adjust_threshold(self):
        total_events = self.buffer_insert_count + self.buffer_waste_count + self.buffer_hit_count
        if total_events < self.adjust_interval:
            return
        total_resolved = self.buffer_hit_count + self.buffer_waste_count
        if total_resolved == 0:
            self._reset_counters()
            return
        waste_ratio = self.buffer_waste_count / total_resolved
        if waste_ratio > 0.5:
            self.min_insert_threshold = min(self.min_insert_threshold * 2, self.MAX_THRESHOLD)
        elif waste_ratio < 0.1 and self.buffer_mem_manager.can_use_mem_size > self.buffer_mem_manager.size * 0.3:
            self.min_insert_threshold = max(self.min_insert_threshold // 2, self.MIN_THRESHOLD)
        self._reset_counters()

    def _reset_counters(self):
        self.buffer_insert_count = 0
        self.buffer_hit_count = 0
        self.buffer_waste_count = 0
