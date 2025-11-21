import torch
import numpy as np
import collections
import xxhash
import threading
import time
from typing import Tuple, Dict, Set, List, Optional, Union
from typing_extensions import override
from sortedcontainers import SortedSet
from .shared_arr import SharedArray
from .radix_cache import UniqueTimeIdGenerator, TreeNode
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

time_gen = UniqueTimeIdGenerator()


class HybridTreeNode(TreeNode):
    def __init__(self):
        super().__init__()
        self.children_list: List[HybridTreeNode] = []
        self.buffer_idx = None
        
        self.depth = 0

    # DONE
    def add_and_return_new_child(self, token_id_key, token_mem_index_value, buffer_idx):
        child = HybridTreeNode()
        child.token_id_key = token_id_key
        child.token_mem_index_value = token_mem_index_value
        child.buffer_idx = buffer_idx
        self.children_list.append(child)
        child.parent = self

        new_len = len(child.token_mem_index_value)
        child.node_value_len = new_len
        child.node_prefix_total_len = self.node_prefix_total_len + new_len
        child.depth = self.depth + 1
        return child

    @override
    def remove_child(self, child_node: "HybridTreeNode"):
        self.children_list.remove(child_node)
        child_node.parent = None
        return

    @override
    def get_compare_key(self):
        return (0 if self.ref_counter == 0 else 1, len(self.children), self.time_id)

    @override
    def is_leaf(self):
        return len(self.children_list) == 0

    def get_buffer_compare_key(self):
        return (self.time_id)

class HybridRadixCache:

    def __init__(self, unique_name, total_token_num, rank_in_node, mem_manager=None, _skip_type_check=False):
        from lightllm.models.qwen3next.mem_manager import Qwen3NextMemoryManager
        if not _skip_type_check:
            assert isinstance(mem_manager, Qwen3NextMemoryManager), "HybridRadixCache only support Qwen3NextMemoryManager."
        self.mem_manager = mem_manager

        self._key_dtype = torch.int64
        self._value_dtype = torch.int64

        self.root_node = HybridTreeNode()
        self.root_node.token_id_key = torch.zeros((0,), device="cpu", dtype=self._key_dtype)
        self.root_node.token_mem_index_value = torch.zeros((0,), device="cpu", dtype=self._value_dtype)
        self.root_node.ref_counter = 1  # 初始化为 1 保证永远不会被 evict 掉

        self.evict_token_tree_set: Set[HybridTreeNode] = SortedSet(key=lambda x: x.get_compare_key())  
        self.evict_buffer_tree_set: Set[HybridTreeNode] = SortedSet(key=lambda x: x.get_buffer_compare_key()) 

        self.refed_tokens_num = SharedArray(f"{unique_name}_refed_tokens_num_{rank_in_node}", (1,), dtype=np.int64)
        self.refed_tokens_num.arr[0] = 0
        self.tree_total_tokens_num = SharedArray(
            f"{unique_name}_tree_total_tokens_num_{rank_in_node}", (1,), dtype=np.int64
        )
        self.tree_total_tokens_num.arr[0] = 0
        self.tree_total_buffer_num = SharedArray(
            f"{unique_name}_tree_total_buffer_num_{rank_in_node}", (1,), dtype=np.int64
        )
        self.tree_total_buffer_num.arr[0] = 0
        
        # Statistics for match_prefix (lock-free counters for performance)
        # In CPython, simple integer operations are atomic due to GIL
        self.match_prefix_total_key_len = 0  # Sum of all input key lengths
        self.match_prefix_total_match_len = 0  # Sum of all matched lengths (len(value))
        self.match_prefix_lock = threading.Lock()  # Only used when reading/resetting in background thread
        
        # Start background thread for periodic statistics printing
        self._stats_thread = threading.Thread(target=self._print_stats_periodically, daemon=True)
        self._stats_thread.start()
        
        logger.info(f"HybridRadixCache initialized!")

    def insert(self, key, value, buffer_idx) -> Tuple[int, Optional[HybridTreeNode]]:
        assert len(key) == len(value) and len(key) >= 1
        assert buffer_idx is not None, "buffer_idx must not be None"
        return self._insert_helper(self.root_node, key, value, buffer_idx)

    def _insert_helper(self, node: HybridTreeNode, key, value, buffer_idx) -> Tuple[int, Optional[HybridTreeNode]]:
        handle_stack = collections.deque()
        update_list = collections.deque()
        handle_stack.append((node, key, value))

        ans_prefix_len = 0
        ans_node = None

        while len(handle_stack) != 0:
            node, key, value = handle_stack.popleft()
            ans_tuple = self._insert_helper_no_recursion(node=node, key=key, value=value, buffer_idx=buffer_idx)
            if len(ans_tuple) == 4:
                (_prefix_len, new_node, new_key, new_value) = ans_tuple
                ans_prefix_len += _prefix_len
                handle_stack.append((new_node, new_key, new_value))
                update_list.append(new_node)  # 确保子节点也会被更新
            else:
                _prefix_len, ans_node = ans_tuple
                ans_prefix_len += _prefix_len
                if ans_node != node:  # 如果返回的节点不是当前节点，说明是新创建的或者已有的子节点
                    update_list.append(ans_node)

            update_list.append(node)

        # 使用集合去重，避免重复处理同一个节点
        unique_nodes = set(update_list)
        for cur_node in unique_nodes:
            # 从集合中移除（如果存在）以便重新添加更新后的排序
            self.evict_token_tree_set.discard(cur_node)
            self.evict_buffer_tree_set.discard(cur_node)
            
            cur_node.update_time()
            
            # 根据条件重新添加到淘汰集合
            if cur_node != self.root_node:
                if cur_node.ref_counter == 0 and cur_node.is_leaf():
                    if cur_node.buffer_idx is None:
                        # Immediately evict - no buffer, no references, leaf node
                        self._immediate_evict_node(cur_node)
                    else:
                        # Has buffer, can be evicted later when needed
                        self.evict_token_tree_set.add(cur_node)
                
                # All nodes with buffers can be evicted (regardless of ref_counter)
                if cur_node.buffer_idx is not None:
                    self.evict_buffer_tree_set.add(cur_node)

        assert ans_node is not None
        return ans_prefix_len, ans_node
    
    def _insert_helper_no_recursion(
        self, node: HybridTreeNode, key: torch.Tensor, value: torch.Tensor, buffer_idx: int
    ) -> Union[Tuple[int, Optional[HybridTreeNode]], Tuple[int, HybridTreeNode, torch.Tensor, torch.Tensor, int]]:
        
        for child in node.children_list:
            if child.node_value_len <= len(key):
                if torch.equal(key[:child.node_value_len], child.token_id_key):
                    if child.node_value_len == len(key):
                        # 更新 buffer_idx
                        if child.buffer_idx is not None:
                            if buffer_idx != child.buffer_idx:
                                # 替换已有的 buffer
                                self.mem_manager.free_state_cache_buffer([child.buffer_idx])
                                child.buffer_idx = buffer_idx
                        else:
                            # 从无 buffer 变为有 buffer
                            child.buffer_idx = buffer_idx
                            self.tree_total_buffer_num.arr[0] += 1
                        return child.node_value_len, child
                    else:
                        return (child.node_value_len, child, key[child.node_value_len:], value[child.node_value_len:])
        
        # No matching prefix found, create new node with the entire key
        new_node = node.add_and_return_new_child(key, value, buffer_idx)
        self.tree_total_tokens_num.arr[0] += len(new_node.token_mem_index_value)
        self.tree_total_buffer_num.arr[0] += 1
        return 0, new_node

    def match_prefix(
        self, key, update_refs=False
    ) -> Tuple[Optional[HybridTreeNode], int, Optional[torch.Tensor], Optional[int]]:
        assert len(key) != 0
        ans_value_list = []
        tree_node = self._match_prefix_helper(self.root_node, key, ans_value_list, update_refs=update_refs)
        
        # Update statistics (lock-free increment for performance)
        key_len = len(key)
        self.match_prefix_total_key_len += key_len
        
        if tree_node != self.root_node:
            if len(ans_value_list) != 0:
                value = torch.concat(ans_value_list)
            else:
                value = torch.zeros((0,), device="cpu", dtype=self._value_dtype)
            match_len = len(value)
            self.match_prefix_total_match_len += match_len
            return tree_node, match_len, value
        else:
            return None, 0, None
    
    def _match_prefix_helper(
        self, node: HybridTreeNode, key: torch.Tensor, ans_value_list: list, update_refs=False
    ) -> HybridTreeNode:
        handle_stack = collections.deque()
        update_list = collections.deque()
        ref_update_list = collections.deque()  # 记录需要更新引用的节点
        handle_stack.append((node, key))

        ans_node = None

        while len(handle_stack) != 0:
            node, key = handle_stack.popleft()
            ans_tuple = self._match_prefix_helper_no_recursion(
                node=node, key=key, ans_value_list=ans_value_list, update_refs=False  # 不在这里更新refs
            )
            if isinstance(ans_tuple, tuple):
                new_node, new_key = ans_tuple
                handle_stack.append((new_node, new_key))
                update_list.append(new_node)  # 确保子节点也会被更新
                if update_refs:
                    ref_update_list.append(new_node)  # 记录需要更新引用的节点
            else:
                ans_node = ans_tuple
                if ans_node != node:  # 如果返回的节点不是当前节点
                    update_list.append(ans_node)
                    if update_refs:
                        ref_update_list.append(ans_node)

            update_list.append(node)
            if update_refs and node != self.root_node:
                ref_update_list.append(node)

        # 使用集合去重，避免重复处理同一个节点
        unique_nodes = set(update_list)
        unique_ref_nodes = set(ref_update_list)
        
        # 第一步：从淘汰集合中移除所有需要更新的节点
        for cur_node in unique_nodes:
            self.evict_token_tree_set.discard(cur_node)
            self.evict_buffer_tree_set.discard(cur_node)
        
        # 第二步：更新引用计数（此时节点已经从sorted set中移除）
        if update_refs:
            for cur_node in unique_ref_nodes:
                if cur_node != self.root_node:
                    cur_node.ref_counter += 1
                    # from 0 to 1 need update refs token num
                    if cur_node.ref_counter == 1:
                        self.refed_tokens_num.arr[0] += len(cur_node.token_mem_index_value)
        
        # 第三步：更新时间戳并重新添加到淘汰集合
        for cur_node in unique_nodes:
            cur_node.update_time()
            
            # 根据条件重新添加到淘汰集合
            if cur_node != self.root_node:
                if cur_node.ref_counter == 0 and cur_node.is_leaf():
                    if cur_node.buffer_idx is None:
                        # Immediately evict - no buffer, no references, leaf node
                        self._immediate_evict_node(cur_node)
                    else:
                        # Has buffer, can be evicted later when needed
                        self.evict_token_tree_set.add(cur_node)
                
                # All nodes with buffers can be evicted (regardless of ref_counter)
                if cur_node.buffer_idx is not None:
                    self.evict_buffer_tree_set.add(cur_node)

        return ans_node

    def _match_prefix_helper_no_recursion(
        self, node: HybridTreeNode, key: torch.Tensor, ans_value_list: list, update_refs=False
    ) -> HybridTreeNode:
        # 注意：update_refs 参数已经不在这里处理，而是在 _match_prefix_helper 中处理
        # 这样可以确保在修改 ref_counter 之前先从 sorted set 中移除节点
        
        if len(key) == 0:
            return node

        # Try to find a child whose token_id_key completely matches a prefix of key
        for child in node.children_list:
            child_len = len(child.token_id_key)
            if child_len <= len(key):
                # Check if child's tokens match the prefix of key
                if torch.equal(key[:child_len], child.token_id_key):
                    # Found a matching prefix, add to value list
                    ans_value_list.append(child.token_mem_index_value)
                    return (child, key[child_len:])

        # No matching prefix found, return current node
        return node

    def dec_node_ref_counter(self, node: HybridTreeNode):
        if node is None:
            return
        
        tmp_node = node

        while tmp_node is not None:
            if tmp_node.ref_counter == 1:
                self.refed_tokens_num.arr[0] -= len(tmp_node.token_mem_index_value)
            
            tmp_node.ref_counter -= 1
            
            # 当 ref_counter 从 1 变为 0 时，节点可以被淘汰
            if tmp_node.ref_counter == 0 and tmp_node != self.root_node:
                if tmp_node.buffer_idx is None and tmp_node.is_leaf():
                    # Immediately evict this node - it has no buffer and no references
                    parent_to_check = tmp_node.parent
                    self._immediate_evict_node(tmp_node)
                    # Move to parent for next iteration
                    tmp_node = parent_to_check
                    continue
                else:
                    # Add to appropriate eviction sets
                    if tmp_node.is_leaf():
                        self.evict_token_tree_set.add(tmp_node)
                    if tmp_node.buffer_idx is not None:
                        self.evict_buffer_tree_set.add(tmp_node)
            
            tmp_node = tmp_node.parent
        
        return
        
    def add_node_ref_counter(self, node: HybridTreeNode):
        if node is None:
            return

        tmp_node = node
        while tmp_node is not None:
            # 当 ref_counter 从 0 变为 1 时，需要从淘汰集合中移除
            if tmp_node.ref_counter == 0 and tmp_node != self.root_node:
                self.evict_token_tree_set.discard(tmp_node)
                # Keep in buffer set - buffers can be evicted regardless of ref_counter
                self.refed_tokens_num.arr[0] += len(tmp_node.token_mem_index_value)
            
            tmp_node.ref_counter += 1
            tmp_node = tmp_node.parent

        return

    def free_radix_cache_to_get_enough_token(self, need_token_num, need_buffer_num=0):
        assert self.mem_manager is not None
        
        # 第一步：淘汰 token（只能淘汰叶子节点）
        if need_token_num > self.mem_manager.can_use_mem_size:
            need_evict_token_num = need_token_num - self.mem_manager.can_use_mem_size
            release_mems = []
            release_buffer_idxs = []
            num_evicted_tokens = 0
            
            while num_evicted_tokens < need_evict_token_num:
                assert len(self.evict_token_tree_set) != 0, "No more tokens to evict!"
                node: HybridTreeNode = self.evict_token_tree_set.pop(0)
                
                # 规则1&2: 叶子节点被淘汰时，整个节点都要被淘汰
                assert node.is_leaf(), "Only leaf nodes can be evicted for tokens"
                assert node.ref_counter == 0, "Cannot evict referenced node"
                
                num_evicted_tokens += len(node.token_mem_index_value)
                release_mems.append(node.token_mem_index_value)
                
                # 如果有 buffer，也要释放
                if node.buffer_idx is not None:
                    release_buffer_idxs.append(node.buffer_idx)
                    # 从 buffer 淘汰集合中移除
                    self.evict_buffer_tree_set.discard(node)
                
                self.tree_total_tokens_num.arr[0] -= len(node.token_mem_index_value)
                if node.buffer_idx is not None:
                    self.tree_total_buffer_num.arr[0] -= 1
                
                # 从父节点移除
                parent_node: HybridTreeNode = node.parent
                parent_node.remove_child(node)
                
                # 检查父节点是否变成了叶子节点，如果是且 ref_counter == 0，加入淘汰集合
                if parent_node.is_leaf() and parent_node.ref_counter == 0 and parent_node != self.root_node:
                    self.evict_token_tree_set.add(parent_node)

            if len(release_mems) > 0:
                mem_index = torch.concat(release_mems)
                self.mem_manager.free(mem_index)
            if len(release_buffer_idxs) > 0:
                self.mem_manager.free_state_cache_buffer(release_buffer_idxs)
        
        # 第二步：淘汰 buffer（可以淘汰所有节点的 buffer，除了 root）
        if need_buffer_num > self.mem_manager.get_state_cache_can_use_size():
            need_evict_buffer_num = need_buffer_num - self.mem_manager.get_state_cache_can_use_size()
            release_buffer_idxs = []
            num_evicted_buffers = 0
            
            while num_evicted_buffers < need_evict_buffer_num:
                assert len(self.evict_buffer_tree_set) != 0, "No more buffers to evict!"
                node: HybridTreeNode = self.evict_buffer_tree_set.pop(0)
                
                # Buffers can be evicted regardless of ref_counter
                assert node.buffer_idx is not None, "Node should have buffer_idx"
                
                num_evicted_buffers += 1
                release_buffer_idxs.append(node.buffer_idx)
                node.buffer_idx = None
                self.tree_total_buffer_num.arr[0] -= 1
                
                # Check if node should be immediately evicted (leaf without buffer and ref_counter == 0)
                if node.is_leaf() and node.ref_counter == 0:
                    # Immediately evict this node - it has no buffer and no references
                    self._immediate_evict_node(node)
                # 非叶子节点：只淘汰 buffer，保留节点
            
            if len(release_buffer_idxs) > 0:
                self.mem_manager.free_state_cache_buffer(release_buffer_idxs)
        
        return

    def _immediate_evict_node(self, node: HybridTreeNode):
        """Immediately evict a node that has no buffer and ref_counter == 0"""
        if node == self.root_node:
            return
        
        # Remove from parent
        parent_node = node.parent
        parent_node.remove_child(node)
        
        # Free memory
        self.mem_manager.free(node.token_mem_index_value)
        self.tree_total_tokens_num.arr[0] -= len(node.token_mem_index_value)
        
        # Check if parent became a leaf without buffer
        if parent_node.is_leaf() and parent_node.buffer_idx is None and parent_node.ref_counter == 0:
            self._immediate_evict_node(parent_node)

    def release_no_buffer_leaf_node(self, node: HybridTreeNode, release_mems: List[torch.Tensor]):
        """递归删除没有 buffer 的叶子节点链"""
        if node.is_leaf() and node.buffer_idx is None and node != self.root_node:
            assert node.ref_counter == 0, "Cannot release referenced node"
            
            # 从淘汰集合中移除
            self.evict_token_tree_set.discard(node)
            self.evict_buffer_tree_set.discard(node)
            
            # 释放内存
            release_mems.append(node.token_mem_index_value)
            self.tree_total_tokens_num.arr[0] -= len(node.token_mem_index_value)
            
            # 从父节点移除
            parent_node: HybridTreeNode = node.parent
            parent_node.remove_child(node)
            
            # 检查父节点是否也变成了没有 buffer 的叶子节点
            if parent_node.is_leaf() and parent_node.ref_counter == 0 and parent_node != self.root_node:
                # 如果父节点变成了叶子节点，加入到 token 淘汰集合
                self.evict_token_tree_set.add(parent_node)
            
            # 递归检查父节点
            self.release_no_buffer_leaf_node(parent_node, release_mems)

        return

    def get_refed_tokens_num(self):
        return self.refed_tokens_num.arr[0]

    def get_tree_total_tokens_num(self):
        return self.tree_total_tokens_num.arr[0]
    
    def _print_stats_periodically(self):
        """Background thread to print statistics every 10 seconds"""
        while True:
            time.sleep(10)
            try:
                if self.match_prefix_total_key_len > 0:
                    match_ratio = (self.match_prefix_total_match_len / self.match_prefix_total_key_len) * 100.0
                    logger.info(
                        f"RadixCache Match Ratio: {match_ratio:.2f}%"
                    )
            except Exception as e:
                # Silently ignore errors to avoid affecting main thread
                pass
