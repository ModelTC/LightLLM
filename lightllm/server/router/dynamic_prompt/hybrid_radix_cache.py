from typing import Set, List, Optional

import torch
from sortedcontainers import SortedSet

from lightllm.server.router.dynamic_prompt.radix_cache import RadixCache, TreeNode
from lightllm.common.mamba_cache_mem_manager.cache_manager import MambaCacheManager
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.log_utils import init_logger
from .detached_mamba_checkpoint_cache import DetachedMambaCheckpointCache

logger = init_logger(__name__)


class HybridRadixCache(RadixCache):
    def __init__(self, unique_name, total_token_num, rank_in_node, kv_cache_mem_manager):
        super().__init__(unique_name, total_token_num, rank_in_node, kv_cache_mem_manager)
        assert hasattr(kv_cache_mem_manager, "mamba_cache_mem_manager")
        cpu_mgr = getattr(kv_cache_mem_manager, "cpu_mamba_cache_manager", None)
        if cpu_mgr is not None:
            self.buffer_mem_manager = cpu_mgr
        else:
            self.buffer_mem_manager: MambaCacheManager = kv_cache_mem_manager.mamba_cache_mem_manager
        self.evict_buffer_set: Set[TreeNode] = SortedSet(key=lambda x: x.buffer_time)
        args = get_env_start_args()
        self.detached_mamba_manager = (
            DetachedMambaCheckpointCache(self.buffer_mem_manager, args.cpu_cache_token_page_size)
            if args.enable_cpu_cache
            else None
        )

    def match_prefix(self, key, update_refs=False):
        assert len(key) != 0
        ans_value_list = []
        tree_node = self._match_prefix_helper(self.root_node, key, ans_value_list, update_refs=update_refs)
        unbuffered_prefix_pos = 0
        while tree_node != self.root_node and tree_node.buffer_idx is None:
            unbuffered_prefix_pos += len(ans_value_list[-1]) if ans_value_list else 0

            next_node = tree_node.parent

            if update_refs:
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
            return None, 0, None, unbuffered_prefix_pos

        update_node = tree_node
        while update_node != self.root_node:
            if update_node.buffer_idx is not None:
                self.evict_buffer_set.discard(update_node)
                update_node.update_buffer_time()
                self.evict_buffer_set.add(update_node)
            update_node = update_node.parent

        kv_len = tree_node.node_prefix_total_len
        value = torch.concat(ans_value_list)
        return tree_node, kv_len, value, unbuffered_prefix_pos

    def match_prefix_kv(self, key, update_refs=False):
        return RadixCache.match_prefix(self, key, update_refs=update_refs)

    def insert_with_buffer(self, key, value, unbuffered_prefix_pos, buffer_snapshot_fn):
        """Insert tokens and ensure buffers at the boundary and leaf.

        Args:
            key: full input token ids (int64 tensor)
            value: corresponding KV cache memory indices (int64 tensor)
            unbuffered_prefix_pos: token position of prefix needing a buffer.
                                   0 means no boundary snapshot needed.
            buffer_snapshot_fn: callable(node) -> None, called for nodes needing
                                a buffer. Must check node.buffer_idx before acting.

        Returns:
            (prefix_len, leaf_node)
        """
        if unbuffered_prefix_pos > 0:
            # First insert the prefix portion to ensure a node exists at the boundary
            boundary_key = key[:unbuffered_prefix_pos]
            boundary_value = value[:unbuffered_prefix_pos]
            _, boundary_node = self.insert(boundary_key, boundary_value)
            if boundary_node is not None and boundary_node.buffer_idx is None:
                buffer_snapshot_fn(boundary_node)

        # Insert the full key
        prefix_len, leaf_node = self.insert(key, value)
        if leaf_node is not None and leaf_node.buffer_idx is None:
            buffer_snapshot_fn(leaf_node)

        return prefix_len, leaf_node

    def add_buffer_idx_to_node(self, node: TreeNode, buffer_idx: int):
        """Set buffer_idx for a node and add it to evict_buffer_set."""
        self.evict_buffer_set.discard(node)
        if node.is_leaf():
            self.evict_tree_set.discard(node)
        if node.buffer_idx is not None:
            self.buffer_mem_manager.free([node.buffer_idx])
        node.buffer_idx = buffer_idx
        node.update_buffer_time()
        self.evict_buffer_set.add(node)
        if node.is_leaf():
            self.evict_tree_set.add(node)
        return

    def free_radix_cache_to_get_enough_buffer(self, need_buffer_num, protected_nodes: Optional[Set[TreeNode]] = None):
        if need_buffer_num > self.buffer_mem_manager.can_use_mem_size:
            need_evict_buffer_num = need_buffer_num - self.buffer_mem_manager.can_use_mem_size
            release_buffers = []

            def release_buffer(buffer_idx):
                release_buffers.append(buffer_idx)
                return

            self._evict_buffer(need_evict_buffer_num, release_buffer, protected_nodes=protected_nodes)
            if len(release_buffers) > 0:
                self.buffer_mem_manager.free(release_buffers)
            if need_buffer_num > self.buffer_mem_manager.can_use_mem_size and self.detached_mamba_manager is not None:
                self.detached_mamba_manager.evict_to_get_enough_buffer(
                    need_buffer_num - self.buffer_mem_manager.can_use_mem_size
                )
        return

    def _evict_buffer(
        self,
        need_evict_buffer_num,
        evict_buffer_callback,
        protected_nodes: Optional[Set[TreeNode]] = None,
    ):
        # Two-pass eviction: first evict buffers from unreferenced nodes,
        # then from referenced nodes only if necessary.
        protected_nodes = protected_nodes or set()
        deferred_referenced = []
        deferred_protected = []
        while need_evict_buffer_num > 0:
            if not self.evict_buffer_set:
                break
            node = self.evict_buffer_set.pop(0)
            assert node.buffer_idx is not None

            if node in protected_nodes:
                deferred_protected.append(node)
                continue

            if node.ref_counter > 0:
                deferred_referenced.append(node)
                continue

            evict_buffer_callback(node.buffer_idx)
            node.buffer_idx = None
            need_evict_buffer_num -= 1

            if node.is_leaf() and node.ref_counter == 0:
                self.evict_tree_set.add(node)

        for node in deferred_referenced:
            if need_evict_buffer_num <= 0:
                self.evict_buffer_set.add(node)
                continue

            if node in protected_nodes:
                deferred_protected.append(node)
                continue

            evict_buffer_callback(node.buffer_idx)
            node.buffer_idx = None
            need_evict_buffer_num -= 1

        for node in deferred_protected:
            self.evict_buffer_set.add(node)
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
                if not self._detach_buffer_before_node_removal(node):
                    evict_buffer_callback(node.buffer_idx)
                node.buffer_idx = None
            # update total token num
            self.tree_total_tokens_num.arr[0] -= len(node.token_mem_index_value)
            parent_node: TreeNode = node.parent
            parent_node.remove_child(node)
            if parent_node.is_leaf():
                self.evict_tree_set.add(parent_node)

        return

    def _detach_buffer_before_node_removal(self, node: TreeNode) -> bool:
        if self.detached_mamba_manager is None:
            return False
        if node.node_prefix_total_len % self.detached_mamba_manager.token_page_size != 0:
            return False

        prefix_tokens = self._get_prefix_token_ids(node)
        return self.detached_mamba_manager.add_checkpoint(prefix_tokens, node.buffer_idx)

    def _get_prefix_token_ids(self, node: TreeNode) -> torch.Tensor:
        parts = []
        cur_node = node
        while cur_node != self.root_node:
            parts.append(cur_node.token_id_key)
            cur_node = cur_node.parent

        if not parts:
            return torch.zeros((0,), dtype=self.root_node.token_id_key.dtype)
        return torch.concat(parts[::-1])
