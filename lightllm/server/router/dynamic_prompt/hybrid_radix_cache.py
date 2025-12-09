import torch
import numpy as np
import collections
import xxhash
import threading
import time
from typing import Tuple, Dict, Set, List, Optional, Union
from typing_extensions import override
from sortedcontainers import SortedSet
from abc import ABC, abstractmethod
import math
from dataclasses import dataclass, field

from .shared_arr import SharedArray
from .radix_cache import UniqueTimeIdGenerator
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

time_gen = UniqueTimeIdGenerator()


class HybridRadixNode:
    def __init__(self):
        # Core data
        self.edge: Tuple[int, ...] = ()
        self.childrens_list: List["HybridRadixNode"] = []
        self.parent: Optional["HybridRadixNode"] = None

        # LightLLM specific
        self.token_id_key = torch.zeros((0,), device="cpu", dtype=torch.int64)
        self.token_mem_index_value = torch.zeros((0,), device="cpu", dtype=torch.int64)
        self.buffer_idx: Optional[int] = None

        # Node metadata
        self.node_value_len: int = 0
        self.node_prefix_total_len: int = 0
        self.ref_counter: int = 0
        self.time_id: int = 0

        # Eviction metadata
        self.hit_count: int = 0
        self.insert_time: float = 0.0
        self.last_access: float = 0.0
        self.node_id: int = 0

    def is_leaf(self) -> bool:
        return len(self.childrens_list) == 0

    def has_buffer(self) -> bool:
        return self.buffer_idx is not None

    def is_referenced(self) -> bool:
        return self.ref_counter > 0

    def collect_path_values(self) -> torch.Tensor:
        """Collect all values from root to this node."""
        segments = []
        node = self
        while node.parent is not None:
            if len(node.token_mem_index_value) > 0:
                segments.append(node.token_mem_index_value)
            node = node.parent

        if not segments:
            return torch.zeros((0,), device="cpu", dtype=torch.int64)

        # Reverse order and concatenate
        segments.reverse()
        return torch.cat(segments, dim=0)

    def update_time(self):
        self.time_id = time_gen.generate_time_id()
        self.last_access = time.time()

    def remove_child(self, child_node: "HybridRadixNode"):
        child_node.parent = None
        self.childrens_list.remove(child_node)

    def get_kv_cache_compare_key(self):
        return (self.is_referenced(), not self.is_leaf(), self.has_buffer(), self.time_id)

    def get_buffer_compare_key(self):
        return self.time_id

    def add_and_return_new_child(self, token_id_key, token_mem_index_value, buffer_idx):
        child = HybridRadixNode()
        child.token_id_key = token_id_key
        child.token_mem_index_value = token_mem_index_value
        child.buffer_idx = buffer_idx
        self.childrens_list.append(child)
        child.parent = self

        new_len = len(child.token_mem_index_value)
        child.node_value_len = new_len
        child.node_prefix_total_len = self.node_prefix_total_len + new_len
        return child


class HybridRadixCache:
    def __init__(self, unique_name, total_token_num, rank_in_node, mem_manager=None):
        from lightllm.models.qwen3next.mem_manager import Qwen3NextMemoryManager

        self.mem_manager: Qwen3NextMemoryManager = mem_manager

        self._key_dtype = torch.int64
        self._value_dtype = torch.int64

        self.root_node = HybridRadixNode()
        self.root_node.token_id_key = torch.zeros((0,), device="cpu", dtype=self._key_dtype)
        self.root_node.token_mem_index_value = torch.zeros((0,), device="cpu", dtype=self._value_dtype)
        self.root_node.ref_counter = 1  # 初始化为 1 保证永远不会被 evict 掉

        self.evict_kv_cache_tree_set: Set[HybridRadixNode] = SortedSet(key=lambda x: x.get_kv_cache_compare_key())
        self.evict_buffer_tree_set: Set[HybridRadixNode] = SortedSet(key=lambda x: x.get_buffer_compare_key())
        self.evict_kv_cache_tree_set.add(self.root_node)
        self.evict_buffer_tree_set.add(self.root_node)

        self.refed_tokens_num = SharedArray(f"{unique_name}_refed_tokens_num_{rank_in_node}", (1,), dtype=np.int64)
        self.refed_tokens_num.arr[0] = 0
        self.tree_total_tokens_num = SharedArray(
            f"{unique_name}_tree_total_tokens_num_{rank_in_node}", (1,), dtype=np.int64
        )
        self.tree_total_tokens_num.arr[0] = 0
        self.tree_total_buffers_num = SharedArray(
            f"{unique_name}_tree_total_buffers_num_{rank_in_node}", (1,), dtype=np.int64
        )
        self.tree_total_buffers_num.arr[0] = 0

    def update_evict_info(self, node: HybridRadixNode):
        # Update time once at the beginning
        if node == self.root_node:
            return

        if node.has_buffer():
            # Remove and re-add to update position in sorted set
            try:
                self.evict_buffer_tree_set.discard(node)
            except ValueError:
                pass
        if not node.is_leaf():
            # Remove and re-add to update position in sorted set
            try:
                self.evict_kv_cache_tree_set.discard(node)
            except ValueError:
                pass

        node.update_time()

        if node.has_buffer():
            self.evict_buffer_tree_set.add(node)
        if not node.is_leaf():
            self.evict_kv_cache_tree_set.add(node)
        return

    def insert(self, key, value, buffer_idx: int) -> Tuple[int, Optional[HybridRadixNode]]:
        logger.info(
            f"insert key len: {len(key)}, value len: {len(value)}, buffer_idx: {buffer_idx} key[:10]: {key[:10]}"
        )
        assert key is not None and value is not None and buffer_idx is not None
        assert len(key) == len(value) and len(key) >= 1

        return self._insert_helper(self.root_node, key, value, buffer_idx, len(key), 0)

    def _insert_helper(
        self, node: HybridRadixNode, key, value, buffer_idx, key_len, prefix_len
    ) -> Tuple[int, Optional[HybridRadixNode]]:
        # 插入的前提是已经完全覆盖当前节点
        # 遍历当前的所有子节点，找到第一个完全匹配的节点，继续插入
        # 如果找不到完全匹配的节点，则直接插入
        for child in node.childrens_list:
            if key_len < child.node_value_len:
                continue
            if torch.equal(child.token_id_key, key[0 : child.node_value_len]):
                # 完全匹配，继续向下插入
                return self._insert_helper(
                    child,
                    key[child.node_value_len :],
                    value[child.node_value_len :],
                    buffer_idx,
                    key_len - child.node_value_len,
                    prefix_len + child.node_value_len,
                )

        # 没有找到完全匹配的节点，直接插入
        # Prevent set corruption by removing node before modifying it (which changes is_leaf status)
        if node != self.root_node:
            try:
                self.evict_kv_cache_tree_set.discard(node)
            except ValueError:
                pass
            if node.has_buffer():
                try:
                    self.evict_buffer_tree_set.discard(node)
                except ValueError:
                    pass

        new_child = node.add_and_return_new_child(key, value, buffer_idx)
        new_child.update_time()
        self.evict_kv_cache_tree_set.add(new_child)
        self.evict_buffer_tree_set.add(new_child)
        self.update_evict_info(node)
        self.tree_total_tokens_num.arr[0] += len(value)
        self.tree_total_buffers_num.arr[0] += 1
        return prefix_len, new_child

    def match_prefix(self, key, update_refs=False):
        logger.info(f"match_prefix key len: {len(key)}, update_refs: {update_refs} key[:10]: {key[:10]}")
        if len(key) == 0:
            return None, 0, None

        ans_value_list = []
        tree_node = self._match_prefix_helper(self.root_node, key, ans_value_list, update_refs=update_refs)

        if tree_node != self.root_node and tree_node is not None:
            if len(ans_value_list) != 0:
                value = torch.cat(ans_value_list, dim=0)
            else:
                value = torch.zeros((0,), device="cpu", dtype=self._value_dtype)
            logger.info(f"match_prefix success len: {len(value)}")
            return tree_node, len(value), value
        else:
            logger.info("match_prefix failed")
            return None, 0, None

    def _match_prefix_helper(
        self, node: HybridRadixNode, key, ans_value_list, update_refs=False
    ) -> Optional[HybridRadixNode]:
        # 匹配的前提是已经完全覆盖当前节点
        # 遍历所有节点，假设完全匹配key， 则返回。

        if len(key) == 0:
            return node

        for child in node.childrens_list:
            if len(key) < child.node_value_len:
                continue
            if torch.equal(child.token_id_key, key[0 : child.node_value_len]):
                # 完全匹配，继续向下匹配
                ans_value_list.append(child.token_mem_index_value)
                match_node = self._match_prefix_helper(
                    child,
                    key[child.node_value_len :],
                    ans_value_list,
                    update_refs=update_refs,
                )
                if match_node is not None:
                    if update_refs:
                        self.add_node_ref_counter(child)
                        self.update_evict_info(child)
                    return match_node
                else:
                    ans_value_list.pop()
        return node

    def evict_kv_cache(self, need_remove_tokens, evict_memindexes, evict_buffer_indexes):
        logger.info(
            f"evict_kv_cache need: {need_remove_tokens}"
            f"total: {self.tree_total_tokens_num.arr[0]}"
            f"refed: {self.refed_tokens_num.arr[0]}"
        )
        if self.tree_total_tokens_num.arr[0] - self.refed_tokens_num.arr[0] < need_remove_tokens:
            assert False, f"""can not free tree tokens {need_remove_tokens},
                              tree_total_tokens_num {self.tree_total_tokens_num.arr[0]},
                              refed_tokens_num {self.refed_tokens_num.arr[0]}"""
        num_evicted = 0
        while num_evicted < need_remove_tokens:
            node: HybridRadixNode = self.evict_kv_cache_tree_set.pop(0)
            assert (
                node.ref_counter == 0 and len(node.childrens_list) == 0 and node != self.root_node
            ), "error evict tree node state"
            num_evicted += len(node.token_mem_index_value)
            evict_memindexes.append(node.token_mem_index_value)
            if node.has_buffer():
                evict_buffer_indexes.append(node.buffer_idx)
                self.tree_total_buffers_num.arr[0] -= 1
            # update total token num
            self.tree_total_tokens_num.arr[0] -= len(node.token_mem_index_value)
            parent_node: HybridRadixNode = node.parent

            # Prevent set corruption by removing parent before modifying it
            if parent_node != self.root_node:
                try:
                    self.evict_kv_cache_tree_set.discard(parent_node)
                except ValueError:
                    pass
                if parent_node.has_buffer():
                    try:
                        self.evict_buffer_tree_set.discard(parent_node)
                    except ValueError:
                        pass

            parent_node.remove_child(node)
            self.update_evict_info(parent_node)
        return

    def evict_buffer_cache(self, need_remove_buffers, evict_buffer_indexes):
        if self.tree_total_buffers_num.arr[0] < need_remove_buffers:
            assert False, f"""can not free tree buffers {need_remove_buffers},
                              tree_total_buffers_num {self.tree_total_buffers_num.arr[0]}"""
        num_evicted = 0
        while num_evicted < need_remove_buffers:
            node: HybridRadixNode = self.evict_buffer_tree_set.pop(0)
            assert node.has_buffer() and node != self.root_node, "error evict buffer node state"
            num_evicted += 1
            evict_buffer_indexes.append(node.buffer_idx)
            node.buffer_idx = None
            self.update_evict_info(node)
        return

    def free_radix_cache_to_get_enough_token(self, need_token_num, need_buffer_num=0):
        logger.info(
            f"free_radix_cache need_token: {need_token_num}"
            f"need_buffer: {need_buffer_num}"
            f"can_use: {self.mem_manager.can_use_mem_size}"
            f"state_cache_can_use: {self.mem_manager.get_state_cache_can_use_size()}"
        )
        if need_token_num > self.mem_manager.can_use_mem_size:
            need_evict_token_num = need_token_num - self.mem_manager.can_use_mem_size
            if need_evict_token_num > 0:
                evict_memindexes = []
                evict_buffer_indexes = []
                self.evict_kv_cache(need_evict_token_num, evict_memindexes, evict_buffer_indexes)
                evict_memindexes = torch.concat(evict_memindexes)
                self.mem_manager.free(evict_memindexes)
                self.mem_manager.free_state_cache_buffer(evict_buffer_indexes)

        if need_buffer_num > self.mem_manager.get_state_cache_can_use_size():
            need_evict_buffer_num = need_buffer_num - self.mem_manager.get_state_cache_can_use_size()
            if need_evict_buffer_num > 0:
                evict_buffer_indexes = []
                self.evict_buffer_cache(need_evict_buffer_num, evict_buffer_indexes)
                self.mem_manager.free_state_cache_buffer(evict_buffer_indexes)
        return

    def get_tree_total_tokens_num(self):
        return self.tree_total_tokens_num.arr[0]

    def get_refed_tokens_num(self):
        return self.refed_tokens_num.arr[0]

    def add_node_ref_counter(self, node: HybridRadixNode):
        if node is None:
            return

        while node is not None:
            if node != self.root_node:
                try:
                    self.evict_kv_cache_tree_set.discard(node)
                except ValueError:
                    pass

            if node.ref_counter == 0:
                self.refed_tokens_num.arr[0] += len(node.token_mem_index_value)
            node.ref_counter += 1

            if node != self.root_node:
                self.evict_kv_cache_tree_set.add(node)

            node = node.parent
        return

    def dec_node_ref_counter(self, node: HybridRadixNode):
        if node is None:
            return

        while node is not None:
            if node != self.root_node:
                try:
                    self.evict_kv_cache_tree_set.discard(node)
                except ValueError:
                    pass

            if node.ref_counter == 1:
                self.refed_tokens_num.arr[0] -= len(node.token_mem_index_value)
            node.ref_counter -= 1

            if node != self.root_node:
                self.evict_kv_cache_tree_set.add(node)

            node = node.parent
        return


class _RadixCacheReadOnlyClient:
    """
    router 端只读用的客户端，用于从共享内存中读取树结构中的信息，用于进行prompt cache 的调度估计。
    """

    def __init__(self, unique_name, total_token_num, rank_in_node):
        self.refed_tokens_num = SharedArray(f"{unique_name}_refed_tokens_num_{rank_in_node}", (1,), dtype=np.int64)
        self.tree_total_tokens_num = SharedArray(
            f"{unique_name}_tree_total_tokens_num_{rank_in_node}", (1,), dtype=np.int64
        )

    def get_refed_tokens_num(self):
        return self.refed_tokens_num.arr[0]

    def get_tree_total_tokens_num(self):
        return self.tree_total_tokens_num.arr[0]

    def get_unrefed_tokens_num(self):
        return self.tree_total_tokens_num.arr[0] - self.refed_tokens_num.arr[0]


class RadixCacheReadOnlyClient:
    def __init__(self, unique_name, total_token_num, node_world_size, dp_world_size):
        self.dp_rank_clients: List[_RadixCacheReadOnlyClient] = [
            _RadixCacheReadOnlyClient(unique_name, total_token_num, rank_in_node)
            for rank_in_node in range(0, node_world_size, dp_world_size)
        ]

    def get_refed_tokens_num(self, dp_rank_in_node):
        return self.dp_rank_clients[dp_rank_in_node].get_refed_tokens_num()

    def get_tree_total_tokens_num(self, dp_rank_in_node):
        return self.dp_rank_clients[dp_rank_in_node].get_tree_total_tokens_num()

    def get_unrefed_tokens_num(self, dp_rank_in_node):
        return self.dp_rank_clients[dp_rank_in_node].get_unrefed_tokens_num()
