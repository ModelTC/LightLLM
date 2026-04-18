import torch
import numpy as np
import collections
from typing import Tuple, Dict, Set, List, Optional, Union
from sortedcontainers import SortedSet
from lightllm.common.mamba_cache_mem_manager.linear_att_buffer_manager import LinearAttCacheManager
from .shared_arr import SharedArray
from .radix_cache import UniqueTimeIdGenerator, time_gen, match


class PagedTreeNode:
    def __init__(self):
        # children are keyed by the last ``block_hash`` of each child
        self.children: Dict[int, "PagedTreeNode"] = {}
        self.parent: "PagedTreeNode" = None

        # Hash of the last page in this node (None for the empty root).
        self.last_page_hash: Optional[int] = None
        # Number of pages owned by this node (in [1, big_page_num]); 0 for root.
        self.num_pages: int = 0

        # token-level data for this node; length == num_pages * hash_page_size
        self.token_id_key: torch.Tensor = None
        self.token_mem_index_value: torch.Tensor = None

        self.ref_counter = 0
        self.time_id = time_gen.generate_time_id()

        self.node_value_len = 0
        self.node_prefix_total_len = 0

        # Kept for parity with ``TreeNode`` (used by hybrid attention models).
        self.linear_buffer_idx = None

    def get_compare_key(self):
        return (0 if self.ref_counter == 0 else 1, len(self.children), self.time_id)

    def add_and_return_new_child(
        self,
        token_id_key: torch.Tensor,
        token_mem_index_value: torch.Tensor,
        block_hashs: List[int],
        block_linear_idxs: List[int],
    ) -> "PagedTreeNode":
        assert len(block_hashs) >= 1
        child = PagedTreeNode()
        child.last_page_hash = block_hashs[-1]
        child.linear_buffer_idx = block_linear_idxs[-1]
        child.num_pages = len(block_hashs)
        child.token_id_key = token_id_key
        child.token_mem_index_value = token_mem_index_value

        assert child.last_page_hash not in self.children, "duplicate last block hash in children"
        self.children[child.last_page_hash] = child
        child.parent = self

        new_len = len(child.token_mem_index_value)
        child.node_value_len = new_len
        child.node_prefix_total_len = child.parent.node_prefix_total_len + new_len
        return child

    def remove_child(self, child_node: "PagedTreeNode"):
        del self.children[child_node.last_page_hash]
        child_node.parent = None

    def update_time(self):
        self.time_id = time_gen.generate_time_id()

    def is_leaf(self):
        return len(self.children) == 0


class PagedRadixCache:
    def __init__(
        self,
        unique_name: str,
        total_token_num: int,
        rank_in_node: int,
        hash_page_size: int,
        big_page_num: int,
        kv_cache_mem_manager=None,
        linear_att_cache_manager=None,
    ):
        from lightllm.common.kv_cache_mem_manager import MemoryManager

        assert hash_page_size >= 1, "hash_page_size must be >= 1"
        assert big_page_num >= 1, "big_page_num must be >= 1"

        self.hash_page_size = hash_page_size
        self.big_page_num = big_page_num
        self.big_page_tokens = hash_page_size * big_page_num
        self.total_token_num = total_token_num

        self.mem_manager: MemoryManager = kv_cache_mem_manager
        self._key_dtype = torch.int64
        self._value_dtype = torch.int64

        self.root_node = PagedTreeNode()
        self.root_node.token_id_key = torch.zeros((0,), device="cpu", dtype=self._key_dtype)
        self.root_node.token_mem_index_value = torch.zeros((0,), device="cpu", dtype=self._value_dtype)
        self.root_node.ref_counter = 1  # pinned so root is never evicted

        self._evict_tree_set: Set[PagedTreeNode] = SortedSet(key=lambda x: x.get_compare_key())
        self._evict_tree_set_for_linear_att: Set[PagedTreeNode] = SortedSet(key=lambda x: x.get_compare_key())

        self.refed_tokens_num = SharedArray(f"{unique_name}_refed_tokens_num_{rank_in_node}", (1,), dtype=np.int64)
        self.refed_tokens_num.arr[0] = 0
        self.tree_total_tokens_num = SharedArray(
            f"{unique_name}_tree_total_tokens_num_{rank_in_node}", (1,), dtype=np.int64
        )
        self.tree_total_tokens_num.arr[0] = 0
        self.linear_att_cache_manager: LinearAttCacheManager = linear_att_cache_manager

    def _discard_node(self, node: PagedTreeNode):
        assert node.is_leaf(), "only leaf node can be discarded"
        self._evict_tree_set.discard(node)
        self._evict_tree_set_for_linear_att.discard(node)
        return

    def _add_node(self, node: PagedTreeNode):
        if node is self.root_node:
            return

        assert node.is_leaf(), "only leaf node can be added"
        self._evict_tree_set.add(node)
        if node.linear_buffer_idx is not None and node.ref_counter == 0:
            self._evict_tree_set_for_linear_att.add(node)
        return

    def insert(
        self,
        key: torch.Tensor,
        value: Optional[torch.Tensor] = None,
        block_hashs: Optional[List[int]] = None,
        block_linear_idxs: Optional[List[int]] = None,
    ) -> Tuple[int, Optional[PagedTreeNode]]:
        assert key is not None
        if value is None:
            value = key
        assert len(key) == len(value)
        if block_hashs is None:
            block_hashs = []
        if block_linear_idxs is not None:
            block_linear_idxs = []

        assert (
            len(key) == len(block_hashs) * self.hash_page_size
        ), f"key length {len(key)} does not match block_hashs length {len(block_hashs)} * {self.hash_page_size}"
        assert len(block_hashs) == len(
            block_linear_idxs
        ), f"block_hashs length {len(block_hashs)} does not match block_linear_idxs length {len(block_linear_idxs)}"

        if len(block_hashs) == 0:
            return 0, None

        if len(block_hashs) % self.big_page_num == 0:
            assert all(
                e is None for e in block_linear_idxs
            ), "all block_linear_idxs must be None when block_hashs length is a multiple of big_page_num"
        else:
            # TODO, test stable then to delete this assertion
            assert all(
                e is None for e in block_linear_idxs[:-1]
            ), "only the last block_linear_idx can be non-None, for compatibility with non-paged radix cache"
            assert (
                block_linear_idxs[-1] is not None
            ), "the last block_linear_idx must not be None, for compatibility with non-paged radix cache"

        return self._insert_helper(self.root_node, key, value, block_hashs, block_linear_idxs)

    def _insert_helper(
        self,
        node: PagedTreeNode,
        key: torch.Tensor,
        value: torch.Tensor,
        block_hashs: List[int],
        block_linear_idxs: List[int],
    ) -> Tuple[int, Optional[PagedTreeNode]]:
        if node.is_leaf():
            self._discard_node(node)

        node.update_time()

        try:
            if len(block_hashs) > self.big_page_num:
                take_tokens = self.big_page_num * self.hash_page_size
                if block_hashs[self.big_page_num - 1] in node.children:
                    child = node.children[block_hashs[self.big_page_num - 1]]
                    sub_prefix_len, ans_node = self._insert_helper(
                        child,
                        key[take_tokens:],
                        value[take_tokens:],
                        block_hashs[self.big_page_num :],
                        block_linear_idxs[self.big_page_num :],
                    )
                    return take_tokens + sub_prefix_len, ans_node
                else:
                    new_node = node.add_and_return_new_child(
                        key[:take_tokens],
                        value[:take_tokens],
                        block_hashs[: self.big_page_num],
                        block_linear_idxs[: self.big_page_num],
                    )
                    self.tree_total_tokens_num.arr[0] += take_tokens
                    _, ans_node = self._insert_helper(
                        new_node,
                        key[take_tokens:],
                        value[take_tokens:],
                        block_hashs[self.big_page_num :],
                        block_linear_idxs[self.big_page_num :],
                    )
                    return 0, ans_node
            else:
                take = len(block_hashs)
                take_tokens = take * self.hash_page_size
                if block_hashs[-1] in node.children:
                    child = node.children[block_hashs[-1]]
                    assert child.num_pages == len(
                        block_hashs
                    ), "invariant broken: child num_pages does not match block_hashs length"
                    if child.is_leaf():
                        self._discard_node(child)
                    child.update_time()
                    if child.is_leaf():
                        self._add_node(child)

                    if block_linear_idxs[-1] is not None:
                        # 当插入的时候发现，叶节点已经存在的时候，需要把这个叶节点占用的线性缓存释放掉，外部不用处理这个细节了
                        self.linear_att_cache_manager.free_state_cache(free_indexes=[block_linear_idxs[-1]])

                    return take_tokens, child
                else:
                    new_node = node.add_and_return_new_child(
                        key[:take_tokens],
                        value[:take_tokens],
                        block_hashs[:take],
                        block_linear_idxs[:take],
                    )
                    self.tree_total_tokens_num.arr[0] += take_tokens
                    if new_node.is_leaf():
                        self._add_node(new_node)
                    return 0, new_node
        finally:
            if node.is_leaf():
                self._add_node(node)

    def match_prefix(
        self,
        key: torch.Tensor,
        block_hashs: Optional[List[int]] = None,
        update_refs: bool = False,
    ):
        """Look up ``block_hashs`` in the tree.

        Matching only advances into a child when that child's whole
        page range is covered by the caller's ``block_hashs`` (i.e. the
        query aligns with the node's page boundary).  Partial matches of
        a node are never returned.
        """
        assert key is not None, "key must not be None"
        if block_hashs is None:
            block_hashs = []

        assert (
            len(key) == len(block_hashs) * self.hash_page_size
        ), f"key length {len(key)} does not match block_hashs length {len(block_hashs)} * {self.hash_page_size}"

        if len(block_hashs) == 0 and len(key) != 0:
            return None, 0, None

        ans_value_list: List[torch.Tensor] = []
        tree_node = self._match_prefix_helper(
            self.root_node,
            key=key,
            block_hashs=block_hashs,
            ans_value_list=ans_value_list,
            update_refs=update_refs,
        )

        if tree_node is not self.root_node:
            if len(ans_value_list) != 0:
                value = torch.concat(ans_value_list)
            else:
                value = torch.zeros((0,), device="cpu", dtype=self._value_dtype)
            return tree_node, len(value), value
        else:
            if update_refs:
                self.dec_node_ref_counter(self.root_node)
            return None, 0, None

    def _match_prefix_helper(
        self,
        node: PagedTreeNode,
        key: torch.Tensor,
        block_hashs: Optional[List[int]],
        ans_value_list: list,
        update_refs: bool = False,
    ) -> PagedTreeNode:

        if node.is_leaf():
            self._discard_node(node)
        node.update_time()

        try:

            if update_refs:
                node.ref_counter += 1
                # from 0 to 1 need update refs token num
                if node.ref_counter == 1:
                    self.refed_tokens_num.arr[0] += len(node.token_mem_index_value)

            for i in range(min(self.big_page_num, len(block_hashs))):
                page_hash = block_hashs[i]
                if page_hash in node.children:
                    child = node.children[page_hash]
                    assert (
                        child.num_pages == i + 1
                    ), "invariant broken: child num_pages does not match index in block_hashs"
                    ans_value_list.append(child.token_mem_index_value)
                    return self._match_prefix_helper(
                        child,
                        key[(i + 1) * self.hash_page_size :],
                        block_hashs[i + 1 :],
                        ans_value_list,
                        update_refs,
                    )
            return node
        finally:
            if node.is_leaf():
                self._add_node(node)

    def _try_merge(self, child_node: PagedTreeNode) -> Optional[PagedTreeNode]:
        raise NotImplementedError()

    def merge_unreferenced_nodes(self):
        raise NotImplementedError()

    def clear_tree_nodes(self):
        """Only used in tests."""
        self.free_radix_cache_to_get_enough_token(need_token_num=self.total_token_num)
        return

    def dec_node_ref_counter(self, node: PagedTreeNode):
        if node is None:
            return
        old_node = node
        if old_node.is_leaf():
            self._discard_node(old_node)

        while node is not None:
            if node.ref_counter == 1:
                self.refed_tokens_num.arr[0] -= len(node.token_mem_index_value)
            node.ref_counter -= 1
            node = node.parent

        if old_node.is_leaf():
            self._add_node(old_node)
        return

    def add_node_ref_counter(self, node: PagedTreeNode):
        if node is None:
            return
        old_node = node
        if old_node.is_leaf():
            self._discard_node(old_node)

        while node is not None:
            if node.ref_counter == 0:
                self.refed_tokens_num.arr[0] += len(node.token_mem_index_value)
            node.ref_counter += 1
            node = node.parent

        if old_node.is_leaf():
            self._add_node(old_node)
        return

    def get_mem_index_value_by_node(self, node: PagedTreeNode) -> Optional[torch.Tensor]:
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

    def _print_helper(self, node: PagedTreeNode, indent):
        print(
            " " * indent,
            f"num_pages: {node.num_pages} last_hash: {node.last_page_hash} "
            f"k: {node.token_id_key[0:10] if node.token_id_key is not None else None} "
            f"v: {node.token_mem_index_value[0:10] if node.token_mem_index_value is not None else None} "
            f"refs: {node.ref_counter} time_id: {node.time_id} "
            f"prefix_total_len: {node.node_prefix_total_len} "
            f"node_value_len: {node.node_value_len} buffer_idx: {node.buffer_idx}",
        )
        for _, child in node.children.items():
            self._print_helper(child, indent=indent + 2)
        return

    def free_radix_cache_to_get_enough_token(self, need_token_num):
        assert self.mem_manager is not None
        if need_token_num > self.mem_manager.can_use_mem_size:
            need_evict_token_num = need_token_num - self.mem_manager.can_use_mem_size
            release_mems = []
            linear_att_buffer_indexes = []

            def release_mem(mem_index, linear_att_buffer_index):
                release_mems.append(mem_index)
                linear_att_buffer_indexes.append(linear_att_buffer_index)
                return

            self._evict(need_evict_token_num, release_mem)
            mem_index = torch.concat(release_mems)
            self.mem_manager.free(mem_index)
            linear_att_buffer_indexes = [idx for idx in linear_att_buffer_indexes if idx is not None]
            if len(linear_att_buffer_indexes) > 0:
                self.linear_att_cache_manager.free_state_cache(linear_att_buffer_indexes)
        return

    def free_one_linear_buffer(self):
        if self.linear_att_cache_manager is None:
            return
        if self.linear_att_cache_manager.get_free_cache_num() > 0:
            return
        if len(self._evict_tree_set_for_linear_att) == 0:
            return

        node: PagedTreeNode = self._evict_tree_set_for_linear_att.pop(0)
        self._evict_tree_set.discard(node)

        assert (
            node.ref_counter == 0 and len(node.children) == 0 and node is not self.root_node
        ), "error evict tree node state"

        self.tree_total_tokens_num.arr[0] -= len(node.token_mem_index_value)
        self.linear_att_cache_manager.free_state_cache(free_indexes=[node.linear_buffer_idx])
        self.mem_manager.free(node.token_mem_index_value)

        parent_node: PagedTreeNode = node.parent
        parent_node.remove_child(node)
        if parent_node.is_leaf():
            self._add_node(parent_node)
        return

    def _evict(self, need_remove_tokens, evict_callback):
        if self.tree_total_tokens_num.arr[0] - self.refed_tokens_num.arr[0] < need_remove_tokens:
            assert False, f"""can not free tree tokens {need_remove_tokens},
                              tree_total_tokens_num {self.tree_total_tokens_num.arr[0]},
                              refed_tokens_num {self.refed_tokens_num.arr[0]}"""
        num_evicted = 0
        while num_evicted < need_remove_tokens:
            node: PagedTreeNode = self._evict_tree_set.pop(0)
            self._evict_tree_set_for_linear_att.discard(node)

            assert (
                node.ref_counter == 0 and len(node.children) == 0 and node is not self.root_node
            ), "error evict tree node state"
            num_evicted += len(node.token_mem_index_value)
            evict_callback(node.token_mem_index_value, node.linear_buffer_idx)
            self.tree_total_tokens_num.arr[0] -= len(node.token_mem_index_value)
            parent_node: PagedTreeNode = node.parent
            parent_node.remove_child(node)
            if parent_node.is_leaf():
                self._add_node(parent_node)

        return
