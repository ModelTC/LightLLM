# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/router/radix_cache.py
import torch
import heapq
import time
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import Tuple
from sortedcontainers import SortedSet
from .shared_arr import SharedArray, SharedTreeInfoNode, SharedLinkedListManager


class UniqueTimeIdGenerator:
    def __init__(self):
        self.counter = 0

    def generate_time_id(self):
        self.counter += 1
        return self.counter


time_gen = UniqueTimeIdGenerator()


class TreeNode:
    def __init__(self, shared_idx_manager):
        self.shared_idx_manager: SharedLinkedListManager = shared_idx_manager
        self.children = {}  # 这里的键 为 token_id_key 的第一个元素
        self.parent: TreeNode = None
        self.token_id_key: torch.Tensor = None
        self.token_mem_index_value: torch.Tensor = None  # 用于记录存储的 token_index 为每个元素在 token mem 中的index位置
        self.ref_counter = 0
        self.shared_idx_node: SharedTreeInfoNode = self.shared_idx_manager.alloc()
        self.time_id = time_gen.generate_time_id()  # 用于标识时间周期

    def get_compare_key(self):
        return (0 if self.ref_counter == 0 else 1, len(self.children), self.time_id)

    def split_node(self, prefix_len):
        split_parent_node = TreeNode(self.shared_idx_manager)
        split_parent_node.parent = self.parent
        split_parent_node.parent.children[self.token_id_key[0].item()] = split_parent_node
        split_parent_node.token_id_key = self.token_id_key[0:prefix_len]
        split_parent_node.token_mem_index_value = self.token_mem_index_value[0:prefix_len]
        split_parent_node.children = {}
        split_parent_node.children[self.token_id_key[prefix_len].item()] = self
        split_parent_node.ref_counter = self.ref_counter

        split_parent_node.shared_idx_node.set_parent_idx(self.shared_idx_node.get_parent_idx())
        new_len = len(split_parent_node.token_mem_index_value)
        split_parent_node.shared_idx_node.set_node_value_len(new_len)
        split_parent_node.shared_idx_node.set_node_prefix_total_len(
            split_parent_node.get_parent_prefix_total_len() + new_len
        )

        self.token_id_key = self.token_id_key[prefix_len:]
        self.token_mem_index_value = self.token_mem_index_value[prefix_len:]
        self.parent = split_parent_node
        self.shared_idx_node.set_parent_idx(split_parent_node.shared_idx_node.get_idx())
        new_len = len(self.token_mem_index_value)
        self.shared_idx_node.set_node_value_len(new_len)
        self.shared_idx_node.set_node_prefix_total_len(self.get_parent_prefix_total_len() + new_len)

        return split_parent_node

    def add_and_return_new_child(self, token_id_key, token_mem_index_value):
        child = TreeNode(self.shared_idx_manager)
        child.token_id_key = token_id_key
        child.token_mem_index_value = token_mem_index_value
        first_token_key = child.token_id_key[0].item()
        assert first_token_key not in self.children.keys()
        self.children[first_token_key] = child
        child.parent = self

        # 更新shared 信息
        child.shared_idx_node.set_parent_idx(self.shared_idx_node.get_idx())
        new_len = len(child.token_mem_index_value)
        child.shared_idx_node.set_node_value_len(new_len)
        child.shared_idx_node.set_node_prefix_total_len(child.get_parent_prefix_total_len() + new_len)
        return child

    def remove_child(self, child_node):
        del self.children[child_node.token_id_key[0].item()]
        child_node.parent = None
        return

    def update_time(self):
        self.time_id = time_gen.generate_time_id()

    def is_leaf(self):
        return len(self.children) == 0

    def get_parent_prefix_total_len(self):
        return self.parent.shared_idx_node.get_node_prefix_total_len()


def match(key, seq):
    i = 0
    for k, w in zip(key, seq):
        if k != w:
            break
        i += 1
    return i


class RadixCache:
    """
    unique_name 主要用于解决单机，多实列部署时的shm冲突
    """

    def __init__(self, unique_name, total_token_num, tp_id):
        self._key_dtype = torch.int64
        self._value_dtype = torch.int64

        self.shared_idx_manager = SharedLinkedListManager(unique_name, total_token_num, tp_id)

        self.root_node = TreeNode(self.shared_idx_manager)
        self.root_node.token_id_key = torch.zeros((0,), device="cpu", dtype=self._key_dtype)
        self.root_node.token_mem_index_value = torch.zeros((0,), device="cpu", dtype=self._value_dtype)
        self.root_node.ref_counter = 1  # 初始化为 1 保证永远不会被 evict 掉

        self.evict_tree_set = SortedSet(key=lambda x: x.get_compare_key())  # 自定义比较器
        self.evict_tree_set.add(self.root_node)

        self.refed_tokens_num = SharedArray(f"{unique_name}_refed_tokens_num_{tp_id}", (1,), dtype=np.int64)
        self.refed_tokens_num.arr[0] = 0
        self.tree_total_tokens_num = SharedArray(f"{unique_name}_tree_total_tokens_num_{tp_id}", (1,), dtype=np.int64)
        self.tree_total_tokens_num.arr[0] = 0

    def insert(self, key, value=None):
        if value is None:
            value = key

        assert len(key) == len(value) and len(key) >= 1
        return self._insert_helper(self.root_node, key, value)

    def _insert_helper(self, node: TreeNode, key, value):
        if node.is_leaf():
            self.evict_tree_set.discard(node)

        try:
            first_key_id = key[0].item()
            if first_key_id in node.children.keys():
                child: TreeNode = node.children[first_key_id]
                prefix_len = match(key, child.token_id_key)
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
                    return prefix_len
                elif prefix_len < len(key) and prefix_len == len(child.token_id_key):
                    return prefix_len + self._insert_helper(child, key[prefix_len:], value[prefix_len:])
                else:
                    assert False, "can not run to here"

            else:
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

            first_key_id = key[0].item()
            if first_key_id not in node.children.keys():
                return node
            else:
                child = node.children[first_key_id]
                prefix_len = match(key, child.token_id_key)
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

            # 回收 shared 链表资源
            self.shared_idx_manager.free(node.shared_idx_node.get_idx())
        return

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

                self.shared_idx_manager.free(node.shared_idx_node.get_idx())
            else:
                break

        self.tree_total_tokens_num.arr[0] = 0
        self.refed_tokens_num.arr[0] = 0
        return

    def dec_node_ref_counter(self, node):
        while node is not None:
            if node.ref_counter == 1:
                self.refed_tokens_num.arr[0] -= len(node.token_mem_index_value)
            node.ref_counter -= 1
            node = node.parent
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
            f"shared_idx: {node.shared_idx_node.get_idx()} p_idx: {node.shared_idx_node.get_parent_idx()} \
            k: {node.token_id_key[0:10]} v: {node.token_mem_index_value[0:10]} refs: {node.ref_counter} \
            time_id: {node.time_id} prefix_total_len: {node.shared_idx_node.get_node_prefix_total_len()} \
            node_value_len: {node.shared_idx_node.get_node_value_len()}",
        )
        for _, child in node.children.items():
            self._print_helper(child, indent=indent + 2)
        return


class RadixCacheReadOnlyClient:
    """
    router 端只读用的客户端，用于从共享内存中读取树结构中的信息，用于进行prompt cache 的调度估计。
    """

    def __init__(self, unique_name, total_token_num, tp_id):
        self.shared_idx_manager = SharedLinkedListManager(unique_name, total_token_num, tp_id)
        self.refed_tokens_num = SharedArray(f"{unique_name}_refed_tokens_num_{tp_id}", (1,), dtype=np.int64)
        self.tree_total_tokens_num = SharedArray(f"{unique_name}_tree_total_tokens_num_{tp_id}", (1,), dtype=np.int64)

    def get_refed_tokens_num(self):
        return self.refed_tokens_num.arr[0]

    def get_tree_total_tokens_num(self):
        return self.tree_total_tokens_num.arr[0]

    def get_shared_node(self, idx):
        return self.shared_idx_manager.get_shared_node(idx)

    def get_all_parent_shared_nodes(self, idx):
        node = self.shared_idx_manager.get_shared_node(idx)
        ans_list = [node]
        while node.get_parent_idx() != -1:
            node = self.shared_idx_manager.get_shared_node(node.get_parent_idx())
            ans_list.append(node)
        return ans_list


# ///////////////////////////////////////////////////////////////////////////////

if __name__ == "__main__":
    # test 1
    def test1():
        tree = RadixCache("unique_name", 100, 0)
        ans = tree.insert(torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.int64, device="cpu"))
        assert ans == 0
        tree.print_self()
        ans = tree.insert(torch.tensor([0, 1, 2, 3, 4, 7, 8, 9], dtype=torch.int64, device="cpu"))
        assert ans == 5
        tree.print_self()
        ans = tree.insert(torch.tensor([0, 1, 2, 3, 4, 7, 8, 9], dtype=torch.int64, device="cpu"))
        assert ans == 8
        tree.print_self()

        assert tree.get_refed_tokens_num() == 0
        assert tree.get_tree_total_tokens_num() == 13

        # print("evict")
        tree.evict(9, lambda x: x)
        tree.print_self()
        assert tree.get_refed_tokens_num() == 0 and tree.get_tree_total_tokens_num() == 0

    test1()

    # test 2
    def test2():
        tree = RadixCache("unique_name", 100, 1)
        ans = tree.insert(torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.int64, device="cpu"))
        ans = tree.insert(torch.tensor([0, 1, 2, 3, 4, 7, 8, 9], dtype=torch.int64, device="cpu"))
        tree.print_self()

        tree_node, size, values = tree.match_prefix(
            torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64, device="cpu"), update_refs=False
        )
        assert tree_node.shared_idx_node.get_node_prefix_total_len() == 5 and size == 5 and len(values) == 5
        tree_node, size, values = tree.match_prefix(
            torch.tensor([0, 1, 2, 3, 4, 9], dtype=torch.int64, device="cpu"), update_refs=False
        )
        assert tree_node.shared_idx_node.get_node_prefix_total_len() == 5 and size == 5 and len(values) == 5
        tree_node, size, values = tree.match_prefix(
            torch.tensor([0, 1, 2, 3, 4, 7, 8], dtype=torch.int64, device="cpu"), update_refs=False
        )
        assert tree_node.shared_idx_node.get_node_prefix_total_len() == 7 and size == 7 and len(values) == 7
        tree_node, size, values = tree.match_prefix(
            torch.tensor([0, 1, 2, 3, 4, 7, 9], dtype=torch.int64, device="cpu"), update_refs=False
        )
        assert tree_node.shared_idx_node.get_node_prefix_total_len() == 6 and size == 6 and len(values) == 6
        print(ans)
        return

    # test2()

    # test 3
    def test3():
        tree = RadixCache("unique_name", 100, 2)
        ans = tree.insert(torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.int64, device="cpu"))
        ans = tree.insert(torch.tensor([0, 1, 2, 3, 4, 7, 8, 9], dtype=torch.int64, device="cpu"))
        tree.print_self()

        tree_node, size, values = tree.match_prefix(
            torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64, device="cpu"), update_refs=True
        )
        assert tree_node.shared_idx_node.get_node_prefix_total_len() == 5 and size == 5 and len(values) == 5
        assert tree.get_refed_tokens_num() == 5 and tree.get_tree_total_tokens_num() == 13

        tree_node, size, values = tree.match_prefix(
            torch.tensor([0, 1, 2, 3, 4, 7, 9], dtype=torch.int64, device="cpu"), update_refs=True
        )
        assert tree_node.shared_idx_node.get_node_prefix_total_len() == 6 and size == 6 and len(values) == 6
        assert tree.get_refed_tokens_num() == 6 and tree.get_tree_total_tokens_num() == 13

        tree.print_self()
        tree.evict(2, lambda x: x)
        assert tree.get_refed_tokens_num() == 6 and tree.get_tree_total_tokens_num() == 8
        tree.print_self()

        tree.dec_node_ref_counter(tree_node)
        tree.print_self()
        print(ans)
        return

    test3()

    def test4():

        tree = RadixCache("unique_name", 100, 2)
        ans = tree.insert(torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.int64, device="cpu"))
        ans = tree.insert(torch.tensor([0, 1, 2, 3, 4, 7, 8, 9], dtype=torch.int64, device="cpu"))
        tree.print_self()

        tree.clear_tree_nodes()
        assert tree.shared_idx_manager.can_alloc_num() == 100
        print(ans)
        return

    test4()
