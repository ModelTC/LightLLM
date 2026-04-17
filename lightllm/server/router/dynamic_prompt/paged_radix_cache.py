# Paged radix cache, adapted from radix_cache.py in the same directory.
#
# The tree is organized by page hashes instead of individual tokens. Two
# control parameters shape the tree:
#   - ``hash_page_size``: number of tokens that make up one page.
#   - ``big_page_num``  : number of consecutive pages that make up one
#                         "big page" (the mandatory size of any internal
#                         node).
#
# Structural invariants maintained by this class:
#   1. Every non-leaf (internal) node stores exactly ``big_page_num``
#      pages, i.e. ``big_page_num * hash_page_size`` tokens.
#   2. A leaf node may store either ``big_page_num`` pages (a "big leaf")
#      or 1 .. (big_page_num - 1) pages (a "small leaf").
#
# These invariants restrict when the tree can be split:
#   - Small leaves cannot carry children, so when a small leaf needs to
#     gain a descendant it is first grown (page-by-page) up to
#     ``big_page_num`` pages.
#   - Splitting a node in the middle of its page range is forbidden,
#     because it would produce an internal node with fewer than
#     ``big_page_num`` pages.  When a caller's ``block_hashs`` diverges
#     inside an existing node the traversal simply stops at the last
#     fully-matched node.  (Because ``block_hashs`` are content-addressed
#     this is expected to be rare.)
#
# All original ``RadixCache`` interfaces are preserved; ``insert`` and
# ``match_prefix`` gain a ``block_hashs`` argument that drives the paged
# matching logic, while the rest of the bookkeeping (ref counting,
# eviction, shared arrays, read-only client, ...) behaves exactly as in
# ``radix_cache.py``.

import torch
import numpy as np
import collections
from typing import Tuple, Dict, Set, List, Optional, Union
from sortedcontainers import SortedSet

from .shared_arr import SharedArray
from .radix_cache import UniqueTimeIdGenerator, time_gen, match


class PagedTreeNode:
    """A node in the paged radix cache.

    A node is always either a "big leaf" (exactly ``big_page_num``
    pages) or a "small leaf" (1 .. big_page_num - 1 pages).  Because
    block hashes are content-addressed, the hash of the *last* page is
    sufficient to identify the whole page range uniquely; it therefore
    serves both as the value we verify when matching and as the key the
    parent uses to store this node in its ``children`` dict.
    """

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
        self.buffer_idx = None

    def get_compare_key(self):
        return (0 if self.ref_counter == 0 else 1, len(self.children), self.time_id)

    def add_and_return_new_child(
        self,
        token_id_key: torch.Tensor,
        token_mem_index_value: torch.Tensor,
        block_hashs: List[int],
    ) -> "PagedTreeNode":
        assert len(block_hashs) >= 1
        child = PagedTreeNode()
        child.last_page_hash = block_hashs[-1]
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
    """
    Page-aware radix cache.

    Parameters
    ----------
    unique_name : str
        Shared memory identifier (same semantics as in ``RadixCache``).
    total_token_num : int
        Total number of tokens managed (kept for interface parity).
    rank_in_node : int
        Rank of this process (kept for shared-array naming).
    hash_page_size : int
        Number of tokens per page.
    big_page_num : int
        Number of pages per "big page".  Internal nodes must contain
        exactly ``big_page_num`` pages.
    kv_cache_mem_manager : optional
        Underlying memory manager (same as ``RadixCache``).
    """

    def __init__(
        self,
        unique_name: str,
        total_token_num: int,
        rank_in_node: int,
        hash_page_size: int,
        big_page_num: int,
        kv_cache_mem_manager=None,
    ):
        from lightllm.common.kv_cache_mem_manager import MemoryManager

        assert hash_page_size >= 1, "hash_page_size must be >= 1"
        assert big_page_num >= 1, "big_page_num must be >= 1"

        self.hash_page_size = hash_page_size
        self.big_page_num = big_page_num
        self.big_page_tokens = hash_page_size * big_page_num

        self.mem_manager: MemoryManager = kv_cache_mem_manager
        self._key_dtype = torch.int64
        self._value_dtype = torch.int64

        self.root_node = PagedTreeNode()
        self.root_node.token_id_key = torch.zeros((0,), device="cpu", dtype=self._key_dtype)
        self.root_node.token_mem_index_value = torch.zeros((0,), device="cpu", dtype=self._value_dtype)
        self.root_node.ref_counter = 1  # pinned so root is never evicted

        self.evict_tree_set: Set[PagedTreeNode] = SortedSet(key=lambda x: x.get_compare_key())
        self.evict_tree_set.add(self.root_node)

        self.refed_tokens_num = SharedArray(f"{unique_name}_refed_tokens_num_{rank_in_node}", (1,), dtype=np.int64)
        self.refed_tokens_num.arr[0] = 0
        self.tree_total_tokens_num = SharedArray(
            f"{unique_name}_tree_total_tokens_num_{rank_in_node}", (1,), dtype=np.int64
        )
        self.tree_total_tokens_num.arr[0] = 0

    def insert(
        self,
        key: torch.Tensor,
        value: Optional[torch.Tensor] = None,
        block_hashs: Optional[List[int]] = None,
    ) -> Tuple[int, Optional[PagedTreeNode]]:
        """Insert ``block_hashs`` worth of pages into the tree.

        ``key`` / ``value`` and ``block_hashs`` may describe different
        numbers of pages -- e.g. the last partial page's hash has not
        been produced yet, or extra hashes were supplied.  We therefore
        truncate both sides down to the common, page-aligned prefix
        before inserting anything.
        """
        if value is None:
            value = key
        assert len(key) == len(value)

        if block_hashs is None or len(block_hashs) == 0 or len(key) == 0:
            return 0, None

        # Align ``key``/``value`` and ``block_hashs`` to the shortest
        # common page-aligned prefix of the two inputs.
        hash_block_num = min(len(block_hashs), len(key) // self.hash_page_size)
        if hash_block_num == 0:
            return 0, None

        needed_tokens = hash_block_num * self.hash_page_size
        key = key[:needed_tokens]
        value = value[:needed_tokens]
        block_hashs = list(block_hashs[:hash_block_num])

        return self._insert_helper(self.root_node, key, value, block_hashs)

    def _insert_helper(
        self,
        node: PagedTreeNode,
        key: torch.Tensor,
        value: torch.Tensor,
        block_hashs: List[int],
    ) -> Tuple[int, Optional[PagedTreeNode]]:
        if node.is_leaf():
            self.evict_tree_set.discard(node)

        try:
            # No matching child -- append a fresh leaf.  The number of
            # remaining ``block_hashs`` tells us which kind of leaf to
            # create, and we branch explicitly on that.
            if len(block_hashs) > self.big_page_num:
                # ----- big-leaf branch -----
                # Take exactly ``big_page_num`` pages.  More hashes may
                # still remain, so recurse into the new node to keep
                # placing them underneath.
                take_tokens = self.big_page_num * self.hash_page_size
                if block_hashs[self.big_page_num - 1] in node.children:
                    child = node.children[block_hashs[self.big_page_num - 1]]
                    sub_prefix_len, ans_node = self._insert_helper(
                        child, key[take_tokens:], value[take_tokens:], block_hashs[self.big_page_num :]
                    )
                    return take_tokens + sub_prefix_len, ans_node
                else:
                    new_node = node.add_and_return_new_child(
                        key[:take_tokens], value[:take_tokens], block_hashs[: self.big_page_num]
                    )
                    self.tree_total_tokens_num.arr[0] += take_tokens
                    _, ans_node = self._insert_helper(
                        new_node, key[take_tokens:], value[take_tokens:], block_hashs[self.big_page_num :]
                    )
                    return 0, ans_node
            else:

                take = len(block_hashs)
                take_tokens = take * self.hash_page_size
                if block_hashs[-1] in node.children:
                    child = node.children[block_hashs[-1]]
                    return take_tokens, child
                else:
                    new_node = node.add_and_return_new_child(
                        key[:take_tokens],
                        value[:take_tokens],
                        block_hashs[:take],
                    )
                    self.tree_total_tokens_num.arr[0] += take_tokens
                    if new_node.is_leaf():
                        self.evict_tree_set.add(new_node)
                    return 0, new_node
        finally:
            node.update_time()
            if node.is_leaf():
                self.evict_tree_set.add(node)

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
        if block_hashs is None or len(block_hashs) == 0:
            return None, 0, None

        hash_block_num = len(block_hashs)
        needed_tokens = hash_block_num * self.hash_page_size
        assert (
            len(key) >= needed_tokens
        ), f"key length {len(key)} smaller than hash_block_num * hash_page_size = {needed_tokens}"

        key = key[:needed_tokens]

        ans_value_list: List[torch.Tensor] = []
        tree_node = self._match_prefix_helper(
            self.root_node, key, ans_value_list, update_refs=update_refs, block_hashs=list(block_hashs)
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
        ans_value_list: list,
        update_refs: bool = False,
        block_hashs: Optional[List[int]] = None,
    ) -> PagedTreeNode:
        touched: List[PagedTreeNode] = []
        try:
            while True:
                if node.is_leaf():
                    self.evict_tree_set.discard(node)
                touched.append(node)

                if update_refs:
                    node.ref_counter += 1
                    # from 0 to 1 need update refs token num
                    if node.ref_counter == 1:
                        self.refed_tokens_num.arr[0] += len(node.token_mem_index_value)

                if block_hashs is None or len(block_hashs) == 0:
                    return node

                # Children are keyed by ``last_page_hash``, so look up in
                # O(1) the two possible candidates: a big-leaf child at
                # ``block_hashs[big_page_num - 1]`` and, if the query is
                # shorter than a big page, the small-leaf candidate at
                # ``block_hashs[-1]``.  Matching must end at a node
                # boundary, so any mismatch / short query stops here.
                matched_child: Optional[PagedTreeNode] = None
                if len(block_hashs) >= self.big_page_num:
                    matched_child = node.children.get(block_hashs[self.big_page_num - 1])
                    if matched_child is not None and matched_child.num_pages != self.big_page_num:
                        matched_child = None
                if matched_child is None and len(block_hashs) < self.big_page_num:
                    cand = node.children.get(block_hashs[-1])
                    if cand is not None and cand.num_pages == len(block_hashs):
                        matched_child = cand

                if matched_child is None:
                    return node

                ans_value_list.append(matched_child.token_mem_index_value)
                consumed = matched_child.num_pages * self.hash_page_size
                node = matched_child
                key = key[consumed:]
                block_hashs = block_hashs[matched_child.num_pages :]
        finally:
            for n in touched:
                n.update_time()
                if n.is_leaf():
                    self.evict_tree_set.add(n)

    def evict(self, need_remove_tokens, evict_callback):
        if self.tree_total_tokens_num.arr[0] - self.refed_tokens_num.arr[0] < need_remove_tokens:
            assert False, f"""can not free tree tokens {need_remove_tokens},
                              tree_total_tokens_num {self.tree_total_tokens_num.arr[0]},
                              refed_tokens_num {self.refed_tokens_num.arr[0]}"""
        num_evicted = 0
        while num_evicted < need_remove_tokens:
            node: PagedTreeNode = self.evict_tree_set.pop(0)
            assert (
                node.ref_counter == 0 and len(node.children) == 0 and node is not self.root_node
            ), "error evict tree node state"
            num_evicted += len(node.token_mem_index_value)
            evict_callback(node.token_mem_index_value)
            self.tree_total_tokens_num.arr[0] -= len(node.token_mem_index_value)
            parent_node: PagedTreeNode = node.parent
            parent_node.remove_child(node)
            if parent_node.is_leaf():
                self.evict_tree_set.add(parent_node)

        return

    def _try_merge(self, child_node: PagedTreeNode) -> Optional[PagedTreeNode]:
        """
        Merging a parent into its only child is *never* valid under the
        paged invariants: the parent is either the root or an internal
        node carrying exactly ``big_page_num`` pages, so the merged node
        would exceed ``big_page_num`` pages.  The method is kept for API
        parity and always reports "nothing to merge".
        """
        parent_node = child_node.parent
        if (
            parent_node is None
            or parent_node is self.root_node
            or parent_node.ref_counter != 0
            or len(parent_node.children) != 1
            or child_node.ref_counter != 0
            or parent_node.buffer_idx is not None
        ):
            return None
        # Merging would break the "internal nodes are exactly big_page_num pages" rule.
        return None

    def merge_unreferenced_nodes(self):
        # See ``_try_merge`` -- under the paged invariants no merge is legal.
        return

    def assert_leafs_is_right(self):
        for node in self.evict_tree_set:
            if node.is_leaf() and node.ref_counter == 0:
                a = node.token_mem_index_value.cuda()
                assert (self.mem_manager.mem_state[a] == 1).sum().item() == len(a)

    def clear_tree_nodes(self):
        """Only used in tests."""
        while True:
            node: PagedTreeNode = self.evict_tree_set.pop(0)
            if node is not self.root_node:
                parent_node: PagedTreeNode = node.parent
                parent_node.remove_child(node)
                if parent_node.is_leaf():
                    self.evict_tree_set.add(parent_node)
            else:
                break

        self.tree_total_tokens_num.arr[0] = 0
        self.refed_tokens_num.arr[0] = 0
        return

    def dec_node_ref_counter(self, node: PagedTreeNode):
        if node is None:
            return
        old_node = node
        if old_node.is_leaf():
            self.evict_tree_set.discard(old_node)

        while node is not None:
            if node.ref_counter == 1:
                self.refed_tokens_num.arr[0] -= len(node.token_mem_index_value)
            node.ref_counter -= 1
            node = node.parent

        if old_node.is_leaf():
            self.evict_tree_set.add(old_node)
        return

    def add_node_ref_counter(self, node: PagedTreeNode):
        if node is None:
            return
        old_node = node
        if old_node.is_leaf():
            self.evict_tree_set.discard(old_node)

        while node is not None:
            if node.ref_counter == 0:
                self.refed_tokens_num.arr[0] += len(node.token_mem_index_value)
            node.ref_counter += 1
            node = node.parent

        if old_node.is_leaf():
            self.evict_tree_set.add(old_node)
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

            def release_mem(mem_index):
                release_mems.append(mem_index)
                return

            self.evict(need_evict_token_num, release_mem)
            mem_index = torch.concat(release_mems)
            self.mem_manager.free(mem_index)
        return


class _PagedRadixCacheReadOnlyClient:
    """Read-only client mirroring ``_RadixCacheReadOnlyClient`` for the paged cache."""

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


class PagedRadixCacheReadOnlyClient:
    def __init__(self, unique_name, total_token_num, node_world_size, dp_world_size):
        self.dp_rank_clients: List[_PagedRadixCacheReadOnlyClient] = [
            _PagedRadixCacheReadOnlyClient(unique_name, total_token_num, rank_in_node)
            for rank_in_node in range(0, node_world_size, dp_world_size)
        ]

    def get_refed_tokens_num(self, dp_rank_in_node):
        return self.dp_rank_clients[dp_rank_in_node].get_refed_tokens_num()

    def get_tree_total_tokens_num(self, dp_rank_in_node):
        return self.dp_rank_clients[dp_rank_in_node].get_tree_total_tokens_num()

    def get_unrefed_tokens_num(self, dp_rank_in_node):
        return self.dp_rank_clients[dp_rank_in_node].get_unrefed_tokens_num()
