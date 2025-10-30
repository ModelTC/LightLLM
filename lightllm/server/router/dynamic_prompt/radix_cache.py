# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/router/radix_cache.py
import torch
import numpy as np
import collections
from typing import Tuple, Dict, Set, List, Optional, Union
from sortedcontainers import SortedSet
from .shared_arr import SharedArray


class UniqueTimeIdGenerator:
    def __init__(self):
        self.counter = 0

    def generate_time_id(self):
        self.counter += 1
        return self.counter


time_gen = UniqueTimeIdGenerator()


class TreeNode:
    def __init__(self):
        self.children: Dict[int, TreeNode] = {}  # 这里的键 为 token_id_key 的第一个元素
        self.parent: TreeNode = None
        self.token_id_key: torch.Tensor = None
        self.token_mem_index_value: torch.Tensor = None  # 用于记录存储的 token_index 为每个元素在 token mem 中的index位置
        self.ref_counter = 0
        self.time_id = time_gen.generate_time_id()  # 用于标识时间周期

        self.node_value_len = 0
        self.node_prefix_total_len = 0

    def get_compare_key(self):
        return (0 if self.ref_counter == 0 else 1, len(self.children), self.time_id)

    def split_node(self, prefix_len):
        split_parent_node = TreeNode()
        split_parent_node.parent = self.parent
        split_parent_node.parent.children[self.token_id_key[0].item()] = split_parent_node
        split_parent_node.token_id_key = self.token_id_key[0:prefix_len]
        split_parent_node.token_mem_index_value = self.token_mem_index_value[0:prefix_len]
        split_parent_node.children = {}
        split_parent_node.children[self.token_id_key[prefix_len].item()] = self
        split_parent_node.ref_counter = self.ref_counter

        new_len = len(split_parent_node.token_mem_index_value)
        split_parent_node.node_value_len = new_len
        split_parent_node.node_prefix_total_len = split_parent_node.parent.node_prefix_total_len + new_len

        self.token_id_key = self.token_id_key[prefix_len:]
        self.token_mem_index_value = self.token_mem_index_value[prefix_len:]
        self.parent = split_parent_node
        new_len = len(self.token_mem_index_value)
        self.node_value_len = new_len
        self.node_prefix_total_len = self.parent.node_prefix_total_len + new_len
        return split_parent_node

    def add_and_return_new_child(self, token_id_key, token_mem_index_value):
        child = TreeNode()
        child.token_id_key = token_id_key
        child.token_mem_index_value = token_mem_index_value
        first_token_key = child.token_id_key[0].item()
        assert first_token_key not in self.children.keys()
        self.children[first_token_key] = child
        child.parent = self

        new_len = len(child.token_mem_index_value)
        child.node_value_len = new_len
        child.node_prefix_total_len = child.parent.node_prefix_total_len + new_len
        return child

    def remove_child(self, child_node: "TreeNode"):
        del self.children[child_node.token_id_key[0].item()]
        child_node.parent = None
        return

    def update_time(self):
        self.time_id = time_gen.generate_time_id()

    def is_leaf(self):
        return len(self.children) == 0


def match(t1: torch.Tensor, t2: torch.Tensor) -> int:
    # Ensure same shape for comparison: flatten and get min length
    t1_flat = t1.flatten()
    t2_flat = t2.flatten()
    min_len = min(t1_flat.size(0), t2_flat.size(0))

    # Compare elements and find first mismatch
    diff = t1_flat[:min_len] != t2_flat[:min_len]
    mismatch_indices = torch.nonzero(diff)

    if mismatch_indices.numel() == 0:
        return min_len  # All matched up to min_len
    else:
        return mismatch_indices[0].item()


class RadixCache:
    """
    unique_name 主要用于解决单机，多实列部署时的shm冲突
    """

    def __init__(self, unique_name, total_token_num, rank_in_node, mem_manager=None):
        from lightllm.common.mem_manager import MemoryManager

        self.mem_manager: MemoryManager = mem_manager
        self._key_dtype = torch.int64
        self._value_dtype = torch.int64

        self.root_node = TreeNode()
        self.root_node.token_id_key = torch.zeros((0,), device="cpu", dtype=self._key_dtype)
        self.root_node.token_mem_index_value = torch.zeros((0,), device="cpu", dtype=self._value_dtype)
        self.root_node.ref_counter = 1  # 初始化为 1 保证永远不会被 evict 掉

        self.evict_tree_set: Set[TreeNode] = SortedSet(key=lambda x: x.get_compare_key())  # 自定义比较器
        self.evict_tree_set.add(self.root_node)

        self.refed_tokens_num = SharedArray(f"{unique_name}_refed_tokens_num_{rank_in_node}", (1,), dtype=np.int64)
        self.refed_tokens_num.arr[0] = 0
        self.tree_total_tokens_num = SharedArray(
            f"{unique_name}_tree_total_tokens_num_{rank_in_node}", (1,), dtype=np.int64
        )
        self.tree_total_tokens_num.arr[0] = 0

    def insert(self, key, value=None) -> Tuple[int, Optional[TreeNode]]:
        if value is None:
            value = key

        assert len(key) == len(value)  # and len(key) >= 1
        if len(key) == 0:
            return 0, None
        return self._insert_helper(self.root_node, key, value)

    def _insert_helper(self, node: TreeNode, key, value) -> Tuple[int, Optional[TreeNode]]:
        handle_stack = collections.deque()
        update_list = collections.deque()
        handle_stack.append((node, key, value))

        ans_prefix_len = 0
        ans_node = None

        while len(handle_stack) != 0:
            node, key, value = handle_stack.popleft()
            ans_tuple = self._insert_helper_no_recursion(node=node, key=key, value=value)
            if len(ans_tuple) == 4:
                (_prefix_len, new_node, new_key, new_value) = ans_tuple
                ans_prefix_len += _prefix_len
                handle_stack.append((new_node, new_key, new_value))
            else:
                _prefix_len, ans_node = ans_tuple
                ans_prefix_len += _prefix_len

            update_list.append(node)

        while len(update_list) != 0:
            cur_node: TreeNode = update_list.pop()
            cur_node.update_time()
            if cur_node.is_leaf():
                self.evict_tree_set.add(cur_node)

        assert ans_node is not None

        return ans_prefix_len, ans_node

    def _insert_helper_no_recursion(
        self, node: TreeNode, key: torch.Tensor, value: torch.Tensor
    ) -> Union[Tuple[int, Optional[TreeNode]], Tuple[int, TreeNode, torch.Tensor, torch.Tensor]]:
        if node.is_leaf():
            self.evict_tree_set.discard(node)

        first_key_id = key[0].item()
        if first_key_id in node.children.keys():
            child: TreeNode = node.children[first_key_id]
            prefix_len = match(key, child.token_id_key)
            if prefix_len == len(key):
                if prefix_len == len(child.token_id_key):
                    if child.is_leaf():
                        self.evict_tree_set.discard(child)
                    child.update_time()
                    if child.is_leaf():
                        self.evict_tree_set.add(child)
                    return prefix_len, child
                elif prefix_len < len(child.token_id_key):
                    if child.is_leaf():
                        self.evict_tree_set.discard(child)

                    split_parent_node = child.split_node(prefix_len)

                    if split_parent_node.is_leaf():
                        self.evict_tree_set.add(split_parent_node)
                    if child.is_leaf():
                        self.evict_tree_set.add(child)

                    return prefix_len, split_parent_node
                else:
                    assert False, "can not run to here"

            elif prefix_len < len(key) and prefix_len < len(child.token_id_key):
                if child.is_leaf():
                    self.evict_tree_set.discard(child)

                key = key[prefix_len:]
                value = value[prefix_len:]
                split_parent_node = child.split_node(prefix_len)
                new_node = split_parent_node.add_and_return_new_child(key, value)
                # update total token num
                self.tree_total_tokens_num.arr[0] += len(new_node.token_mem_index_value)

                if split_parent_node.is_leaf():
                    self.evict_tree_set.add(split_parent_node)
                if new_node.is_leaf():
                    self.evict_tree_set.add(new_node)

                if child.is_leaf():
                    self.evict_tree_set.add(child)
                return prefix_len, new_node
            elif prefix_len < len(key) and prefix_len == len(child.token_id_key):
                return (prefix_len, child, key[prefix_len:], value[prefix_len:])
            else:
                assert False, "can not run to here"

        else:
            new_node = node.add_and_return_new_child(key, value)
            # update total token num
            self.tree_total_tokens_num.arr[0] += len(new_node.token_mem_index_value)
            if new_node.is_leaf():
                self.evict_tree_set.add(new_node)
            return 0, new_node

    def match_prefix(self, key, update_refs=False):
        assert len(key) != 0
        ans_value_list = []
        tree_node = self._match_prefix_helper(self.root_node, key, ans_value_list, update_refs=update_refs)
        if tree_node != self.root_node:
            if len(ans_value_list) != 0:
                value = torch.concat(ans_value_list)
            else:
                value = torch.zeros((0,), device="cpu", dtype=self._value_dtype)
            return tree_node, len(value), value
        else:
            self.dec_node_ref_counter(self.root_node)
            return None, 0, None

    def _match_prefix_helper(
        self, node: TreeNode, key: torch.Tensor, ans_value_list: list, update_refs=False
    ) -> TreeNode:
        handle_stack = collections.deque()
        update_list = collections.deque()
        handle_stack.append((node, key))

        ans_node = None

        while len(handle_stack) != 0:
            node, key = handle_stack.popleft()
            ans_tuple = self._match_prefix_helper_no_recursion(
                node=node, key=key, ans_value_list=ans_value_list, update_refs=update_refs
            )
            if isinstance(ans_tuple, tuple):
                new_node, new_key = ans_tuple
                handle_stack.append((new_node, new_key))
            else:
                ans_node = ans_tuple

            update_list.append(node)

        while len(update_list) != 0:
            cur_node: TreeNode = update_list.pop()
            cur_node.update_time()
            if cur_node.is_leaf():
                self.evict_tree_set.add(cur_node)

        return ans_node

    def _match_prefix_helper_no_recursion(
        self, node: TreeNode, key: torch.Tensor, ans_value_list: list, update_refs=False
    ) -> TreeNode:
        if node.is_leaf():
            self.evict_tree_set.discard(node)

        if update_refs:
            node.ref_counter += 1
            # from 0 to 1 need update refs token num
            if node.ref_counter == 1:
                self.refed_tokens_num.arr[0] += len(node.token_mem_index_value)

        if len(key) == 0:
            return node

        first_key_id = key[0].item()
        if first_key_id not in node.children.keys():
            return node
        else:
            child = node.children[first_key_id]
            prefix_len = match(key, child.token_id_key)
            if prefix_len == len(child.token_id_key):
                ans_value_list.append(child.token_mem_index_value)
                return (child, key[prefix_len:])
            elif prefix_len < len(child.token_id_key):
                if child.is_leaf():
                    self.evict_tree_set.discard(child)

                split_parent_node = child.split_node(prefix_len)
                ans_value_list.append(split_parent_node.token_mem_index_value)

                if update_refs:
                    split_parent_node.ref_counter += 1
                    # from 0 to 1 need update refs token num
                    if split_parent_node.ref_counter == 1:
                        self.refed_tokens_num.arr[0] += len(split_parent_node.token_mem_index_value)

                if child.is_leaf():
                    self.evict_tree_set.add(child)
                if split_parent_node.is_leaf():
                    self.evict_tree_set.add(split_parent_node)

                return split_parent_node
            else:
                assert False, "error state"

    def evict(self, need_remove_tokens, evict_callback):
        if self.tree_total_tokens_num.arr[0] - self.refed_tokens_num.arr[0] < need_remove_tokens:
            assert False, f"""can not free tree tokens {need_remove_tokens},
                              tree_total_tokens_num {self.tree_total_tokens_num.arr[0]},
                              refed_tokens_num {self.refed_tokens_num.arr[0]}"""
        num_evicted = 0
        while num_evicted < need_remove_tokens:
            node: TreeNode = self.evict_tree_set.pop(0)
            assert (
                node.ref_counter == 0 and len(node.children) == 0 and node != self.root_node
            ), "error evict tree node state"
            num_evicted += len(node.token_mem_index_value)
            evict_callback(node.token_mem_index_value)
            # update total token num
            self.tree_total_tokens_num.arr[0] -= len(node.token_mem_index_value)
            parent_node: TreeNode = node.parent
            parent_node.remove_child(node)
            if parent_node.is_leaf():
                self.evict_tree_set.add(parent_node)

        return

    def _try_merge(self, child_node: TreeNode) -> Optional[TreeNode]:
        """
        合并条件:
        1. 父节点不是根节点。
        2. 父节点的引用计数为 0。
        3. 子节点的引用计数为 0。
        4. 父节点只有一个子节点 (即 child_node)。
        """
        parent_node = child_node.parent
        # 条件检查
        if (
            parent_node is None
            or parent_node == self.root_node
            or parent_node.ref_counter != 0
            or len(parent_node.children) != 1
            or child_node.ref_counter != 0
        ):
            return None

        if child_node.is_leaf():
            self.evict_tree_set.discard(child_node)

        child_node.token_id_key = torch.cat([parent_node.token_id_key, child_node.token_id_key])
        child_node.token_mem_index_value = torch.cat(
            [parent_node.token_mem_index_value, child_node.token_mem_index_value]
        )
        child_node.node_value_len = len(child_node.token_mem_index_value)
        child_node.time_id = max(parent_node.time_id, child_node.time_id)

        grandparent_node = parent_node.parent
        key_in_grandparent = parent_node.token_id_key[0].item()
        grandparent_node.children[key_in_grandparent] = child_node
        child_node.parent = grandparent_node

        parent_node.parent = None

        if child_node.is_leaf():
            self.evict_tree_set.add(child_node)

        return child_node

    def merge_unreferenced_nodes(self):
        worklist = collections.deque(
            [
                node
                for node in self.evict_tree_set
                if node.ref_counter == 0 and node.parent is not None and node.parent != self.root_node
            ]
        )

        while worklist:
            node = worklist.popleft()
            if node.parent is None:
                continue
            merged_node = self._try_merge(node)
            if merged_node:
                worklist.append(merged_node)

    def assert_leafs_is_right(self):
        for node in self.evict_tree_set:
            if node.is_leaf() and node.ref_counter == 0:
                a = node.token_mem_index_value.cuda()
                assert (self.mem_manager.mem_state[a] == 1).sum().item() == len(a)

    def clear_tree_nodes(self):
        """
        该函数只在测试时调用
        """
        while True:
            node: TreeNode = self.evict_tree_set.pop(0)
            if node != self.root_node:
                parent_node: TreeNode = node.parent
                parent_node.remove_child(node)
                if parent_node.is_leaf():
                    self.evict_tree_set.add(parent_node)
            else:
                break

        self.tree_total_tokens_num.arr[0] = 0
        self.refed_tokens_num.arr[0] = 0
        return

    def dec_node_ref_counter(self, node: TreeNode):
        if node is None:
            return
        # 如果减引用的是叶节点，需要先从 evict_tree_set 中移除
        old_node = node
        if old_node.is_leaf():
            self.evict_tree_set.discard(old_node)

        while node is not None:
            if node.ref_counter == 1:
                self.refed_tokens_num.arr[0] -= len(node.token_mem_index_value)
            node.ref_counter -= 1
            node = node.parent

        # 加回。
        if old_node.is_leaf():
            self.evict_tree_set.add(old_node)
        return

    def add_node_ref_counter(self, node: TreeNode):
        if node is None:
            return
        # 如果减引用的是叶节点，需要先从 evict_tree_set 中移除
        old_node = node
        if old_node.is_leaf():
            self.evict_tree_set.discard(old_node)

        while node is not None:
            if node.ref_counter == 0:
                self.refed_tokens_num.arr[0] += len(node.token_mem_index_value)
            node.ref_counter += 1
            node = node.parent

        # 加回。
        if old_node.is_leaf():
            self.evict_tree_set.add(old_node)
        return

    def get_mem_index_value_by_node(self, node: TreeNode) -> Optional[torch.Tensor]:
        if node is None:
            return None

        ans_list = []
        while node is not None:
            ans_list.append(node.token_mem_index_value)
            node = node.parent

        ans_list.reverse()
        return torch.concat(ans_list, dim=0)

    def get_refed_tokens_num(self):
        return self.refed_tokens_num.arr[0]

    def get_tree_total_tokens_num(self):
        return self.tree_total_tokens_num.arr[0]

    def print_self(self, indent=0):
        self._print_helper(self.root_node, indent)

    def _print_helper(self, node: TreeNode, indent):
        print(
            " " * indent,
            f"k: {node.token_id_key[0:10]} v: {node.token_mem_index_value[0:10]} refs: {node.ref_counter} \
            time_id: {node.time_id} prefix_total_len: {node.node_prefix_total_len} \
            node_value_len: {node.node_value_len}",
        )
        for _, child in node.children.items():
            self._print_helper(child, indent=indent + 2)
        return

    def free_radix_cache_to_get_enough_token(self, need_token_num):
        assert self.mem_manager is not None
        if need_token_num > self.mem_manager.can_use_mem_size:
            need_evict_token_num = need_token_num - self.mem_manager.can_use_mem_size
            release_mems = []

            def release_mem(mem_index):
                release_mems.append(mem_index)
                return

            self.evict(need_evict_token_num, release_mem)
            mem_index = torch.concat(release_mems)
            self.mem_manager.free(mem_index)
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
