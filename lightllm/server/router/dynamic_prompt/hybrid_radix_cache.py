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

    def match_prefix(self, key, update_refs=False):
        assert len(key) != 0
        ans_value_list = []
        tree_node = self._match_prefix_helper(self.root_node, key, ans_value_list, update_refs=update_refs)
        evict_token_list = []
        kv_len = tree_node.node_prefix_total_len
        while tree_node != self.root_node and tree_node.buffer_idx is None:
            if tree_node.is_leaf():
                self.evict_tree_set.discard(tree_node)

            # Only update ref_counter when update_refs is True to maintain consistency
            # with _match_prefix_helper which only increments ref_counter when update_refs=True
            if update_refs:
                if tree_node.ref_counter == 1:
                    self.refed_tokens_num.arr[0] -= len(tree_node.token_mem_index_value)
                tree_node.ref_counter -= 1
            kv_len -= len(ans_value_list.pop())
            if tree_node.is_leaf():
                self.evict_tree_set.add(tree_node)
            tree_node = tree_node.parent

        if len(evict_token_list) > 0:
            evict_token_value = torch.concat(evict_token_list)
            self.mem_manager.free(evict_token_value)

        if tree_node == self.root_node:
            return None, kv_len, None

        update_node = tree_node
        while update_node != self.root_node:
            if update_node.buffer_idx is not None:
                self.evict_buffer_set.discard(update_node)
                update_node.update_buffer_time()
                self.evict_buffer_set.add(update_node)
            update_node = update_node.parent

        value = torch.concat(ans_value_list)
        return tree_node, kv_len, value

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
        return

    def free_radix_cache_to_get_enough_buffer(self, need_buffer_num):
        if need_buffer_num > self.buffer_mem_manager.can_use_mem_size:
            need_evict_buffer_num = need_buffer_num - self.buffer_mem_manager.can_use_mem_size
            release_buffers = []

            def release_buffer(buffer_idx):
                release_buffers.append(buffer_idx)
                return

            self._evict_buffer(need_evict_buffer_num, release_buffer)
            if len(release_buffers) > 0:
                self.buffer_mem_manager.free(release_buffers)
        return

    def _evict_buffer(self, need_evict_buffer_num, evict_buffer_callback):
        while need_evict_buffer_num > 0:
            node = self.evict_buffer_set.pop(0)
            assert node.buffer_idx is not None
            evict_buffer_callback(node.buffer_idx)
            node.buffer_idx = None
            need_evict_buffer_num -= 1
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
