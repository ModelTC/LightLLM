# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/router/radix_cache.py
import torch
import numpy as np
import collections
from typing import Any, Tuple, Dict, Set, List, Optional, Union
from sortedcontainers import SortedSet
from .shared_arr import SharedArray
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


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
        self.token_extra_value: Any = None
        self.ref_counter = 0
        self.time_id = time_gen.generate_time_id()  # 用于标识时间周期

        self.node_value_len = 0
        self.node_prefix_total_len = 0

    def get_compare_key(self):
        return (0 if self.ref_counter == 0 else 1, len(self.children), self.time_id)

    def split_node(self, prefix_len, child_key_fn=None, extra_value_ops=None):
        split_parent_node = TreeNode()
        split_parent_node.parent = self.parent
        split_parent_node.parent.children[child_key_fn(self.token_id_key)] = split_parent_node
        split_parent_node.token_id_key = self.token_id_key[0:prefix_len]
        split_parent_node.token_mem_index_value = self.token_mem_index_value[0:prefix_len]
        if self.token_extra_value is not None and extra_value_ops is not None:
            split_parent_node.token_extra_value = extra_value_ops.slice(self.token_extra_value, 0, prefix_len)
            self.token_extra_value = extra_value_ops.slice(self.token_extra_value, prefix_len, len(self.token_id_key))
        split_parent_node.children = {}
        split_parent_node.children[child_key_fn(self.token_id_key[prefix_len:])] = self
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

    def add_and_return_new_child(self, token_id_key, token_mem_index_value, token_extra_value=None, child_key=None):
        child = TreeNode()
        child.token_id_key = token_id_key
        child.token_mem_index_value = token_mem_index_value
        child.token_extra_value = token_extra_value
        first_token_key = child.token_id_key[0].item() if child_key is None else child_key
        assert first_token_key not in self.children.keys()
        self.children[first_token_key] = child
        child.parent = self

        new_len = len(child.token_mem_index_value)
        child.node_value_len = new_len
        child.node_prefix_total_len = child.parent.node_prefix_total_len + new_len
        return child

    def remove_child(self, child_node: "TreeNode"):
        child_key = child_node.token_id_key[0].item()
        if child_key in self.children:
            del self.children[child_key]
            child_node.parent = None
            return
        for key, value in list(self.children.items()):
            if value is child_node:
                del self.children[key]
                child_node.parent = None
                return
        raise KeyError("child node not found")

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

    def __init__(
        self,
        unique_name,
        total_token_num,
        rank_in_node,
        mem_manager=None,
        page_size: int = 1,
        extra_value_ops=None,
    ):
        from lightllm.common.kv_cache_mem_manager import MemoryManager

        self.mem_manager: MemoryManager = mem_manager
        self._key_dtype = torch.int64
        self._value_dtype = torch.int64
        self.page_size = max(1, int(page_size))
        self.extra_value_ops = extra_value_ops

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
        self.swa_tree_total_pages_num = 0
        self.swa_refed_pages_num = 0
        # 每个 prompt-cache 页折算多少 swa 页(DSV4 为 256/128=2);非 swa 场景为 0,_node_swa_pages_num 退化为常数 0。
        self._swa_pages_per_prompt_page = self._probe_swa_pages_per_prompt_page()

    def _probe_swa_pages_per_prompt_page(self) -> int:
        """构造期探测一次 mem_manager 是否带 swa_pool,缓存折算系数,避免热路径反复 getattr。"""
        if self.mem_manager is None or self.extra_value_ops is None:
            return 0
        swa_pool = getattr(self.mem_manager, "swa_pool", None)
        swa_page_size = getattr(swa_pool, "page_size", None)
        if swa_page_size is None:
            return 0
        return (self.page_size + int(swa_page_size) - 1) // int(swa_page_size)

    def _node_swa_pages_num(self, node: TreeNode) -> int:
        if self._swa_pages_per_prompt_page == 0 or node.token_extra_value is None:
            return 0
        valid = node.token_extra_value.swa_page_valid
        if valid is None:
            return 0
        return int(valid.sum().item()) * self._swa_pages_per_prompt_page

    def _align_len(self, length: int) -> int:
        if self.page_size <= 1:
            return int(length)
        return int(length) // self.page_size * self.page_size

    def align_len(self, length: int) -> int:
        return self._align_len(length)

    def _child_key(self, key: torch.Tensor):
        if self.page_size <= 1:
            return key[0].item()
        return tuple(key[: self.page_size].tolist())

    def _match_len(self, key: torch.Tensor, node_key: torch.Tensor) -> int:
        prefix_len = match(key, node_key)
        return self._align_len(prefix_len)

    def _slice_extra(self, extra_value, start: int, end: int):
        if extra_value is None:
            return None
        assert self.extra_value_ops is not None
        return self.extra_value_ops.slice(extra_value, start, end)

    def _concat_extra(self, values: list):
        values = [v for v in values if v is not None]
        if len(values) == 0:
            return None
        assert self.extra_value_ops is not None
        return self.extra_value_ops.concat(values)

    def insert(self, key, value=None, extra_value=None) -> Tuple[int, Optional[TreeNode]]:
        if value is None:
            value = key

        align_len = self._align_len(len(key))
        key = key[:align_len]
        value = value[:align_len]
        if extra_value is not None:
            extra_value = self._slice_extra(extra_value, 0, align_len)

        assert len(key) == len(value)  # and len(key) >= 1
        if len(key) == 0:
            return 0, None
        return self._insert_helper(self.root_node, key, value, extra_value)

    def _insert_helper(self, node: TreeNode, key, value, extra_value) -> Tuple[int, Optional[TreeNode]]:
        handle_stack = collections.deque()
        update_list = collections.deque()
        handle_stack.append((node, key, value, extra_value))

        ans_prefix_len = 0
        ans_node = None

        while len(handle_stack) != 0:
            node, key, value, extra_value = handle_stack.popleft()
            ans_tuple = self._insert_helper_no_recursion(node=node, key=key, value=value, extra_value=extra_value)
            if len(ans_tuple) == 5:
                (_prefix_len, new_node, new_key, new_value, new_extra_value) = ans_tuple
                ans_prefix_len += _prefix_len
                handle_stack.append((new_node, new_key, new_value, new_extra_value))
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
        self, node: TreeNode, key: torch.Tensor, value: torch.Tensor, extra_value=None
    ) -> Union[Tuple[int, Optional[TreeNode]], Tuple[int, TreeNode, torch.Tensor, torch.Tensor, Any]]:
        if node.is_leaf():
            self.evict_tree_set.discard(node)

        first_key_id = self._child_key(key)
        if first_key_id in node.children.keys():
            child: TreeNode = node.children[first_key_id]
            prefix_len = self._match_len(key, child.token_id_key)
            if prefix_len == len(key):
                if prefix_len == len(child.token_id_key):
                    if child.is_leaf():
                        self.evict_tree_set.discard(child)
                    child.update_time()
                    if child.is_leaf():
                        self.evict_tree_set.add(child)
                    return prefix_len, child
                elif prefix_len < len(child.token_id_key):
                    if prefix_len == 0:
                        return 0, node
                    if child.is_leaf():
                        self.evict_tree_set.discard(child)

                    split_parent_node = child.split_node(
                        prefix_len, child_key_fn=self._child_key, extra_value_ops=self.extra_value_ops
                    )

                    if split_parent_node.is_leaf():
                        self.evict_tree_set.add(split_parent_node)
                    if child.is_leaf():
                        self.evict_tree_set.add(child)

                    return prefix_len, split_parent_node
                else:
                    assert False, "can not run to here"

            elif prefix_len < len(key) and prefix_len < len(child.token_id_key):
                if prefix_len == 0:
                    return 0, node
                if child.is_leaf():
                    self.evict_tree_set.discard(child)

                new_extra_value = self._slice_extra(extra_value, prefix_len, len(key))
                key = key[prefix_len:]
                value = value[prefix_len:]
                split_parent_node = child.split_node(
                    prefix_len, child_key_fn=self._child_key, extra_value_ops=self.extra_value_ops
                )
                new_node = split_parent_node.add_and_return_new_child(
                    key,
                    value,
                    token_extra_value=new_extra_value,
                    child_key=self._child_key(key),
                )
                # update total token num
                self.tree_total_tokens_num.arr[0] += len(new_node.token_mem_index_value)
                self.swa_tree_total_pages_num += self._node_swa_pages_num(new_node)

                if split_parent_node.is_leaf():
                    self.evict_tree_set.add(split_parent_node)
                if new_node.is_leaf():
                    self.evict_tree_set.add(new_node)

                if child.is_leaf():
                    self.evict_tree_set.add(child)
                return prefix_len, new_node
            elif prefix_len < len(key) and prefix_len == len(child.token_id_key):
                return (
                    prefix_len,
                    child,
                    key[prefix_len:],
                    value[prefix_len:],
                    self._slice_extra(extra_value, prefix_len, len(key)),
                )
            else:
                assert False, "can not run to here"

        else:
            new_node = node.add_and_return_new_child(
                key,
                value,
                token_extra_value=extra_value,
                child_key=first_key_id,
            )
            # update total token num
            self.tree_total_tokens_num.arr[0] += len(new_node.token_mem_index_value)
            self.swa_tree_total_pages_num += self._node_swa_pages_num(new_node)
            if new_node.is_leaf():
                self.evict_tree_set.add(new_node)
            return 0, new_node

    def match_prefix(self, key, update_refs=False):
        key = key[: self._align_len(len(key))]
        if len(key) == 0:
            return None, 0, None
        key = self._trim_key_by_extra_value_validity(key)
        if len(key) == 0:
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
            if update_refs:
                self.dec_node_ref_counter(self.root_node)
            return None, 0, None

    def _trim_key_by_extra_value_validity(self, key: torch.Tensor) -> torch.Tensor:
        """命中有效性裁剪(extra_value_ops 提供 valid_match_length 时启用,如 DeepSeek-V4 的
        swa 按页 bitmap): 先做一次只读探测遍历得到自然命中与沿路 extra_value,按其有效边界截短
        key,随后的正常遍历(加引用/分裂)只走截短后的前缀 —— 引用计数与最终返回值在同一次遍历
        内保持一致,不存在事后裁剪导致的漏减/多减。

        探测遍历可能分裂部分命中的节点(与正常遍历同语义,树不变式不受影响)。裁剪只会缩短命中,
        没有任何失败路径。"""
        if self.extra_value_ops is None:
            return key
        valid_match_length = getattr(self.extra_value_ops, "valid_match_length", None)
        if valid_match_length is None:
            return key
        probe_values = []
        probe_node = self._match_prefix_helper(self.root_node, key, probe_values, update_refs=False)
        if probe_node == self.root_node or len(probe_values) == 0:
            return key
        natural_len = sum(len(v) for v in probe_values)
        extra_value = self.get_extra_value_by_node(probe_node)
        valid_len = int(valid_match_length(extra_value, natural_len))
        if valid_len < natural_len:
            return key[:valid_len]
        return key

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
                self.swa_refed_pages_num += self._node_swa_pages_num(node)

        if len(key) == 0:
            return node

        first_key_id = self._child_key(key)
        if first_key_id not in node.children.keys():
            return node
        else:
            child = node.children[first_key_id]
            prefix_len = self._match_len(key, child.token_id_key)
            if prefix_len == len(child.token_id_key):
                ans_value_list.append(child.token_mem_index_value)
                return (child, key[prefix_len:])
            elif prefix_len < len(child.token_id_key):
                if prefix_len == 0:
                    return node
                if child.is_leaf():
                    self.evict_tree_set.discard(child)

                split_parent_node = child.split_node(
                    prefix_len, child_key_fn=self._child_key, extra_value_ops=self.extra_value_ops
                )
                ans_value_list.append(split_parent_node.token_mem_index_value)

                if update_refs:
                    split_parent_node.ref_counter += 1
                    # from 0 to 1 need update refs token num
                    if split_parent_node.ref_counter == 1:
                        self.refed_tokens_num.arr[0] += len(split_parent_node.token_mem_index_value)
                        self.swa_refed_pages_num += self._node_swa_pages_num(split_parent_node)

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
            if self.extra_value_ops is not None and node.token_extra_value is not None:
                self.extra_value_ops.free(node.token_extra_value)
            # update total token num
            self.tree_total_tokens_num.arr[0] -= len(node.token_mem_index_value)
            self.swa_tree_total_pages_num -= self._node_swa_pages_num(node)
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
        child_node.token_extra_value = self._concat_extra([parent_node.token_extra_value, child_node.token_extra_value])
        child_node.node_value_len = len(child_node.token_mem_index_value)
        child_node.time_id = max(parent_node.time_id, child_node.time_id)

        grandparent_node = parent_node.parent
        key_in_grandparent = self._child_key(parent_node.token_id_key)
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
        self.swa_tree_total_pages_num = 0
        self.swa_refed_pages_num = 0
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
                self.swa_refed_pages_num -= self._node_swa_pages_num(node)
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
                self.swa_refed_pages_num += self._node_swa_pages_num(node)
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

    def get_extra_value_by_node(self, node: TreeNode):
        if node is None or self.extra_value_ops is None:
            return None

        ans_list = []
        while node is not None:
            if node.token_extra_value is not None:
                ans_list.append(node.token_extra_value)
            node = node.parent

        ans_list.reverse()
        return self._concat_extra(ans_list)

    def get_refed_tokens_num(self):
        return self.refed_tokens_num.arr[0]

    def get_tree_total_tokens_num(self):
        return self.tree_total_tokens_num.arr[0]

    def get_unrefed_swa_pages_num(self):
        return self.swa_tree_total_pages_num - self.swa_refed_pages_num

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

    def free_unreferenced_swa_pages(self, need_pages: int) -> None:
        """DeepSeek-V4 swa free hook: 页 allocator 触底时，回收 ref_count==0 节点的 swa 页。"""
        if self.mem_manager is None or self.extra_value_ops is None:
            return
        allocator = self.mem_manager.swa_page_allocator
        target = allocator.can_use_mem_size + int(need_pages)
        evict_slots = []
        invalidate_payloads = []
        evict_swa_pages = 0
        for free_last in (False, True):
            visited = set()
            for leaf in self.evict_tree_set:
                if allocator.can_use_mem_size + evict_swa_pages >= target:
                    break
                node = leaf
                while node is not None and node is not self.root_node and node.ref_counter == 0:
                    node_id = id(node)
                    if node_id in visited:
                        node = node.parent
                        continue
                    visited.add(node_id)

                    payload = node.token_extra_value
                    if (
                        len(node.token_mem_index_value) > 0
                        and payload is not None
                        and payload.swa_page_valid is not None
                    ):
                        last_page = int(payload.swa_last_valid_page)
                        if last_page >= 0:
                            if free_last:
                                page_slice = slice(last_page, last_page + 1)
                            else:
                                page_slice = slice(0, last_page)
                            valid_pages = int(payload.swa_page_valid[page_slice].sum().item())
                            if valid_pages > 0:
                                start = page_slice.start * self.page_size
                                end = min(page_slice.stop * self.page_size, len(node.token_mem_index_value))
                                if end > start:
                                    evict_slots.append(node.token_mem_index_value[start:end])
                                    invalidate_payloads.append((payload, page_slice, free_last))
                                    evict_swa_pages += valid_pages * self._swa_pages_per_prompt_page
                                    if allocator.can_use_mem_size + evict_swa_pages >= target:
                                        break
                    node = node.parent
            if allocator.can_use_mem_size + evict_swa_pages >= target:
                break
        if len(evict_slots) == 0:
            return
        self.mem_manager.evict_swa(torch.cat(evict_slots))
        for payload, page_slice, free_last in invalidate_payloads:
            payload.swa_page_valid[page_slice] = False
            if free_last:
                payload.swa_last_valid_page = -1
        self.swa_tree_total_pages_num -= evict_swa_pages
        return

    def free_radix_cache_to_get_enough_token(self, need_token_num):
        assert self.mem_manager is not None
        if need_token_num > self.mem_manager.allocator.can_use_mem_size:
            need_evict_token_num = need_token_num - self.mem_manager.allocator.can_use_mem_size
            release_mems = []

            def release_mem(mem_index):
                release_mems.append(mem_index)
                return

            self.evict(need_evict_token_num, release_mem)
            mem_index = torch.concat(release_mems)
            self.mem_manager.free(mem_index)
        return

    def _free_radix_full_nodes_until(self, allocator, need: int) -> None:
        """DeepSeek-V4 压缩池(c4/c128)兑现: 沿 LRU 序逐个驱逐 ref_count==0 的整个 full radix 节点,
        经 mem_manager.free() 级联回收其 c4 页 / c128 槽(evict_c4/evict_c128),每驱逐一个就复查
        *真实* allocator(不靠计数,稳),直到够或已无可驱逐的无引用节点。后者(空闲+可回收仍不足)
        由上游 base_backend admission 的 wait_pause 兜底,allocator 的 assert 是最后防线。"""
        if self.mem_manager is None or allocator is None:
            return
        while allocator.can_use_mem_size < need:
            # 无可驱逐的无引用 token => 停(admission 应已 wait_pause)
            if self.tree_total_tokens_num.arr[0] <= self.refed_tokens_num.arr[0]:
                # 兜底没兜住:admission/realize 估算漂移了。打日志便于定位(否则只会撞下游隐晦的
                # allocator "error alloc state" assert)。
                logger.warning(
                    f"dsv4 compress-pool realize could not free enough: need={need} "
                    f"free={allocator.can_use_mem_size} tree_total={self.tree_total_tokens_num.arr[0]} "
                    f"refed={self.refed_tokens_num.arr[0]} (admission should have paused this req)"
                )
                return
            release_mems = []
            # 复用已测的 evict():弹一个 LRU、ref==0 的叶子(>=1 token),其 full 槽经 free 级联回收压缩槽
            self.evict(1, lambda mem_index: release_mems.append(mem_index))
            self.mem_manager.free(torch.concat(release_mems))
        return

    def free_radix_cache_to_get_enough_c4_pages(self, need_pages: int) -> None:
        allocator = getattr(self.mem_manager, "c4_page_allocator", None) if self.mem_manager is not None else None
        if allocator is None or need_pages <= 0:
            return
        self._free_radix_full_nodes_until(allocator, need_pages)
        return

    def free_radix_cache_to_get_enough_c128_slots(self, need_slots: int) -> None:
        allocator = getattr(self.mem_manager, "c128_allocator", None) if self.mem_manager is not None else None
        if allocator is None or need_slots <= 0:
            return
        self._free_radix_full_nodes_until(allocator, need_slots)
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
