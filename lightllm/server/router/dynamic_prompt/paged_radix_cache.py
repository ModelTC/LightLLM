# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/router/radix_cache.py
import torch
import numpy as np
from typing import Tuple, Dict, Set, List
from sortedcontainers import SortedSet
from .shared_arr import SharedArray
from lightllm.utils.envs_utils import get_page_size


class UniqueTimeIdGenerator:
    def __init__(self):
        self.counter = 0

    def generate_time_id(self):
        self.counter += 1
        return self.counter


time_gen = UniqueTimeIdGenerator()


class TreeNode:
    def __init__(self):
        self.children: Dict[int, TreeNode] = {}  # page_hash -> TreeNode
        self.parent: TreeNode = None
        self.token_id_key: torch.Tensor = None
        self.token_mem_index_value: torch.Tensor = None  # 用于记录存储的 token_index 为每个元素在 token mem 中的index位置
        self.ref_counter = 0
        self.time_id = time_gen.generate_time_id()  # 用于标识时间周期

        self.node_value_len = 0
        self.node_prefix_total_len = 0
        self.total_children_count = 0
        self.page_size = get_page_size()
        self._page_size_is_power_of_2 = (self.page_size & (self.page_size - 1)) == 0
        self._page_size_mask = self.page_size - 1 if self._page_size_is_power_of_2 else None

    def get_compare_key(self):
        return (0 if self.ref_counter == 0 else 1, self.total_children_count, self.time_id)

    def _compute_key(self, tokens: torch.Tensor) -> int:
        page_tokens = tokens[: self.page_size]
        return page_tokens.item() if self.page_size == 1 else hash(page_tokens.cpu().numpy().tobytes())

    def find_matched_child(self, token_id_key: torch.Tensor) -> Tuple["TreeNode", int]:
        target_key = self._compute_key(token_id_key)
        if target_key in self.children:
            child = self.children[target_key]
            prefix_len = match(token_id_key, child.token_id_key)
            # 只匹配page_size的整数倍长度
            if self.page_size > 1:
                if prefix_len % self.page_size != 0:
                    if self._page_size_is_power_of_2:
                        # 位运算加速
                        prefix_len = prefix_len & ~self._page_size_mask
                    else:
                        prefix_len = (prefix_len // self.page_size) * self.page_size
                    if prefix_len == 0:
                        return None, 0
            return child, prefix_len

        return None, 0

    def split_node(self, prefix_len):
        split_parent_node = TreeNode()
        split_parent_node.parent = self.parent
        self.parent.children[self._compute_key(self.token_id_key)] = split_parent_node

        split_parent_node.token_id_key = self.token_id_key[0:prefix_len]
        split_parent_node.token_mem_index_value = self.token_mem_index_value[0:prefix_len]
        split_parent_node.children = {}

        remaining_tokens = self.token_id_key[prefix_len:]
        split_parent_node.children[self._compute_key(remaining_tokens)] = self
        split_parent_node.ref_counter = self.ref_counter
        split_parent_node.total_children_count = 1

        new_len = len(split_parent_node.token_mem_index_value)
        split_parent_node.node_value_len = new_len
        split_parent_node.node_prefix_total_len = split_parent_node.parent.node_prefix_total_len + new_len

        self.token_id_key = remaining_tokens
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

        self.children[self._compute_key(token_id_key)] = child
        child.parent = self
        self.total_children_count += 1

        new_len = len(child.token_mem_index_value)
        child.node_value_len = new_len
        child.node_prefix_total_len = child.parent.node_prefix_total_len + new_len
        return child

    def remove_child(self, child_node: "TreeNode"):
        del self.children[self._compute_key(child_node.token_id_key)]
        child_node.parent = None
        self.total_children_count -= 1
        return

    def update_time(self):
        self.time_id = time_gen.generate_time_id()

    def is_leaf(self):
        return self.total_children_count == 0


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


class PagedRadixCache:
    """
    unique_name 主要用于解决单机，多实列部署时的shm冲突
    """

    def __init__(self, unique_name, total_token_num, rank_in_node, mem_manager=None):
        self.mem_manager = mem_manager
        self._key_dtype = torch.int64
        self._value_dtype = torch.int64
        # 预计算page_size相关的常量
        self.page_size = get_page_size()
        self._page_size_is_power_of_2 = (self.page_size & (self.page_size - 1)) == 0
        self._page_size_mask = self.page_size - 1 if self._page_size_is_power_of_2 else None

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

    def _get_page_aligned_key(self, key, value=None):
        aligned_len = len(key)
        if aligned_len == 0:
            return None, None
        # page_size > 1时, 需要确保输入的key长度是page_size的整数倍
        if self.page_size > 1:
            if aligned_len % self.page_size != 0:
                if self._page_size_is_power_of_2:
                    # 位运算加速
                    aligned_len = aligned_len & ~self._page_size_mask
                else:
                    aligned_len = (aligned_len // self.page_size) * self.page_size
                return (
                    key[:aligned_len] if aligned_len > 0 else None,
                    value[:aligned_len] if value is not None and aligned_len > 0 else None,
                )
        return key, value

    def insert(self, key, value=None):
        if value is None:
            value = key

        assert len(key) == len(value)  # and len(key) >= 1
        key, value = self._get_page_aligned_key(key, value)
        if key is None:
            return 0
        return self._insert_helper(self.root_node, key, value)

    def _insert_helper(self, node: TreeNode, key, value):
        if node.is_leaf():
            self.evict_tree_set.discard(node)

        try:
            child, prefix_len = node.find_matched_child(key)
            if child is not None:
                if prefix_len == len(key):
                    if child.is_leaf():
                        self.evict_tree_set.discard(child)
                    child.update_time()
                    if child.is_leaf():
                        self.evict_tree_set.add(child)
                    return prefix_len
                elif prefix_len < len(key) and prefix_len < len(child.token_id_key):
                    if child.is_leaf():
                        self.evict_tree_set.discard(child)

                    remaining_key = key[prefix_len:]
                    remaining_value = value[prefix_len:]
                    split_parent_node = child.split_node(prefix_len)
                    new_node = split_parent_node.add_and_return_new_child(remaining_key, remaining_value)
                    # update total token num
                    self.tree_total_tokens_num.arr[0] += len(new_node.token_mem_index_value)
                    if new_node.is_leaf():
                        self.evict_tree_set.add(new_node)

                    if split_parent_node.is_leaf():
                        self.evict_tree_set.add(split_parent_node)

                    if child.is_leaf():
                        self.evict_tree_set.add(child)
                    return prefix_len
                elif prefix_len < len(key) and prefix_len == len(child.token_id_key):
                    return prefix_len + self._insert_helper(child, key[prefix_len:], value[prefix_len:])
                else:
                    assert False, "can not run to here"

            new_node = node.add_and_return_new_child(key, value)
            # update total token num
            self.tree_total_tokens_num.arr[0] += len(new_node.token_mem_index_value)
            if new_node.is_leaf():
                self.evict_tree_set.add(new_node)
            return 0
        finally:
            node.update_time()
            if node.is_leaf():
                self.evict_tree_set.add(node)

    def match_prefix(self, key, update_refs=False):
        assert len(key) != 0
        key, _ = self._get_page_aligned_key(key)
        if key is None:
            return None, 0, None

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

    def _match_prefix_helper(self, node: TreeNode, key, ans_value_list: list, update_refs=False) -> TreeNode:
        if node.is_leaf():
            self.evict_tree_set.discard(node)

        if update_refs:
            node.ref_counter += 1
            # from 0 to 1 need update refs token num
            if node.ref_counter == 1:
                self.refed_tokens_num.arr[0] += len(node.token_mem_index_value)

        try:
            if len(key) == 0:
                return node

            child, prefix_len = node.find_matched_child(key)
            if child is not None:
                if prefix_len == len(child.token_id_key):
                    ans_value_list.append(child.token_mem_index_value)
                    return self._match_prefix_helper(child, key[prefix_len:], ans_value_list, update_refs=update_refs)
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

            return node
        finally:
            node.update_time()
            if node.is_leaf():
                self.evict_tree_set.add(node)

    def evict(self, need_remove_tokens, evict_callback):
        if self.tree_total_tokens_num.arr[0] - self.refed_tokens_num.arr[0] < need_remove_tokens:
            assert False, f"""can not free tree tokens {need_remove_tokens},
                              tree_total_tokens_num {self.tree_total_tokens_num.arr[0]},
                              refed_tokens_num {self.refed_tokens_num.arr[0]}"""
        num_evicted = 0
        while num_evicted < need_remove_tokens:
            node: TreeNode = self.evict_tree_set.pop(0)
            assert node.ref_counter == 0 and node.is_leaf() and node != self.root_node, "error evict tree node state"
            num_evicted += len(node.token_mem_index_value)
            evict_callback(node.token_mem_index_value)
            # update total token num
            self.tree_total_tokens_num.arr[0] -= len(node.token_mem_index_value)
            parent_node: TreeNode = node.parent
            parent_node.remove_child(node)
            if parent_node.is_leaf():
                self.evict_tree_set.add(parent_node)

        return

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

    def free_radix_cache_to_get_enough_token(
        self, need_token_num=None, b_seq_len=None, b_ready_cache_len=None, is_prefill=False
    ):
        assert self.mem_manager is not None
        need_pages = 0
        can_use_pages = 0
        if hasattr(self.mem_manager, "can_use_page_size") and self.page_size > 1 and b_seq_len is not None:

            def get_need_page_size(page_size, b_seq_len, b_ready_cache_len=None, is_prefill=False):
                need_new_pages = 0
                if is_prefill:
                    need_tokens_array = b_seq_len - b_ready_cache_len
                    need_pages_array = (need_tokens_array + page_size - 1) // page_size
                    need_new_pages = need_pages_array.sum()
                else:
                    mask = (b_seq_len - 1) % page_size == 0
                    need_new_pages = mask.sum()
                return need_new_pages

            need_pages = get_need_page_size(self.page_size, b_seq_len, b_ready_cache_len, is_prefill)
            can_use_pages = self.mem_manager.can_use_page_size
        if need_token_num > self.mem_manager.can_use_mem_size or need_pages > can_use_pages:
            need_evict_single_token_num = need_token_num - self.mem_manager.can_use_mem_size
            need_evict_page_token_num = (need_pages - can_use_pages) * self.page_size
            need_evict_token_num = max(need_evict_single_token_num, need_evict_page_token_num)
            remaining_tokens = self.get_tree_total_tokens_num() - self.get_refed_tokens_num()
            need_evict_token_num = min(need_evict_token_num, remaining_tokens)
            release_mems = []

            def release_mem(mem_index):
                release_mems.append(mem_index)
                return

            self.evict(need_evict_token_num, release_mem)
            if release_mems:
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
