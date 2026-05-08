import torch
import numpy as np
from typing import Tuple, Dict, Set, List, Optional
from sortedcontainers import SortedSet, SortedDict
from lightllm.common.linear_att_cache_manager import LinearAttCacheManager
from .shared_arr import SharedArray
from .radix_cache import time_gen


class LinearAttPagedTreeNode:
    def __init__(self, hash_page_size: int, big_page_num: int):
        self.hash_page_size = hash_page_size
        self.big_page_num = big_page_num

        # children are keyed by the last ``block_hash`` of each child
        self.children: Dict[int, "LinearAttPagedTreeNode"] = {}
        self.parent: "LinearAttPagedTreeNode" = None

        # Hash of the last page in this node (None for the empty root).
        self.page_num = None  # 页面数量，只能是 1 或者 big_page_num
        self.page_hash: Optional[int] = None

        # token-level data for this node; length == num_pages * hash_page_size
        self.token_id_key: torch.Tensor = None
        self.token_mem_index_value: torch.Tensor = None

        self.ref_counter = 0
        self.time_id = time_gen.generate_time_id()

        self.node_value_len = 0
        self.node_prefix_total_len = 0

        # Kept for parity with ``TreeNode`` (used by hybrid attention models).
        self.small_page_buffer_idx = None  # 这个是对应小页的buffer_id, 可能存在可能不存在
        self.big_page_buffer_idx = None  # 当初始化后，如果该页面是大页，则该buffer_id 必然存在不是None

    def is_big_page_node(self):
        assert self.node_prefix_total_len % self.hash_page_size == 0
        return self.node_prefix_total_len % (self.hash_page_size * self.big_page_num) == 0

    def get_compare_key(self):
        assert len(self.children) == 0
        if self.is_big_page_node():
            keya = 1
        else:
            if self.small_page_buffer_idx is None:
                keya = 0
            else:
                keya = 1
        # 对于叶节点，非大页节点，如果不存在buffer_idx 的时候，说明无法被复用了，所以应该提前被回收掉，放在evict_tree_set的前面。
        return (0 if self.ref_counter == 0 else 1, keya, self.time_id)

    def get_compare_key_for_buffer_idx(self):
        assert self.is_big_page_node() is False
        # 对于有 buffer_id 的节点的回收处理比较器
        assert self.small_page_buffer_idx is not None
        return (self.time_id,)

    def add_and_return_new_child(
        self,
        token_id_key: torch.Tensor,
        token_mem_index_value: torch.Tensor,
        block_hash: int,
        small_page_buffer_idx: Optional[int],
    ) -> "LinearAttPagedTreeNode":
        assert len(token_id_key) == self.hash_page_size == len(token_mem_index_value)
        child = LinearAttPagedTreeNode(hash_page_size=self.hash_page_size, big_page_num=self.big_page_num)
        child.page_hash = block_hash
        child.small_page_buffer_idx = small_page_buffer_idx
        child.token_id_key = token_id_key
        child.token_mem_index_value = token_mem_index_value
        child.page_num = 1
        assert child.page_hash not in self.children, "duplicate last block hash in children"
        self.children[child.page_hash] = child
        child.parent = self

        new_len = len(child.token_mem_index_value)
        child.node_value_len = new_len
        child.node_prefix_total_len = child.parent.node_prefix_total_len + new_len
        return child

    def add_and_return_new_big_page_child(
        self, token_id_key: torch.Tensor, token_mem_index_value: torch.Tensor, block_hash: int, big_page_buffer_idx: int
    ) -> "LinearAttPagedTreeNode":
        assert len(token_id_key) == self.hash_page_size * self.big_page_num == len(token_mem_index_value)
        child = LinearAttPagedTreeNode(hash_page_size=self.hash_page_size, big_page_num=self.big_page_num)
        child.page_hash = block_hash
        child.token_id_key = token_id_key
        child.token_mem_index_value = token_mem_index_value
        child.page_num = self.big_page_num
        child.big_page_buffer_idx = big_page_buffer_idx
        assert child.big_page_buffer_idx is not None
        assert child.page_hash not in self.children, "duplicate last block hash in children"
        child.parent = self
        self.children[child.page_hash] = child
        new_len = len(child.token_mem_index_value)
        child.node_value_len = new_len
        child.node_prefix_total_len = child.parent.node_prefix_total_len + new_len
        return child

    def remove_child(self, child_node: "LinearAttPagedTreeNode"):
        del self.children[child_node.page_hash]
        child_node.parent = None

    def update_time(self):
        self.time_id = time_gen.generate_time_id()

    def is_leaf(self):
        return len(self.children) == 0


class LinearAttPagedRadixCache:
    def __init__(
        self,
        unique_name: str,
        total_token_num: int,
        rank_in_node: int,
        hash_page_size: int,
        big_page_num: int,
        kv_cache_mem_manager=None,
        linear_att_small_page_buffers=None,
    ):
        from lightllm.common.kv_cache_mem_manager import MemoryManager

        assert hash_page_size >= 1, "hash_page_size must be >= 1"
        assert big_page_num >= 1, "big_page_num must be >= 1"

        self.hash_page_size = hash_page_size
        self.big_page_num = big_page_num
        self.big_page_tokens = hash_page_size * big_page_num
        self.total_token_num = total_token_num

        self.mem_manager: MemoryManager = kv_cache_mem_manager

        self.linear_att_big_page_buffers: LinearAttCacheManager = self.mem_manager.linear_att_big_page_buffers
        self._key_dtype = torch.int64
        self._value_dtype = torch.int64

        self.root_node = LinearAttPagedTreeNode(hash_page_size=hash_page_size, big_page_num=big_page_num)
        self.root_node.token_id_key = torch.zeros((0,), device="cpu", dtype=self._key_dtype)
        self.root_node.token_mem_index_value = torch.zeros((0,), device="cpu", dtype=self._value_dtype)
        self.root_node.ref_counter = 1
        self.root_node.page_num = self.big_page_num

        self._evict_tree_set: Set[LinearAttPagedTreeNode] = SortedSet(key=lambda x: x.get_compare_key())
        self._evict_tree_set_for_linear_att: Set[LinearAttPagedTreeNode] = SortedSet(
            key=lambda x: x.get_compare_key_for_buffer_idx()
        )

        self.refed_tokens_num = SharedArray(f"{unique_name}_refed_tokens_num_{rank_in_node}", (1,), dtype=np.int64)
        self.refed_tokens_num.arr[0] = 0
        self.tree_total_tokens_num = SharedArray(
            f"{unique_name}_tree_total_tokens_num_{rank_in_node}", (1,), dtype=np.int64
        )
        self.tree_total_tokens_num.arr[0] = 0
        self.linear_att_small_page_buffers: LinearAttCacheManager = linear_att_small_page_buffers

    def _discard_node(self, node: LinearAttPagedTreeNode):
        if node.is_leaf():
            self._evict_tree_set.discard(node)
        if node.small_page_buffer_idx is not None:
            self._evict_tree_set_for_linear_att.discard(node)
        return

    def _add_node(self, node: LinearAttPagedTreeNode):
        if node.is_leaf():
            self._evict_tree_set.add(node)
        if node.small_page_buffer_idx is not None:
            self._evict_tree_set_for_linear_att.add(node)
        return

    def insert(
        self,
        key: torch.Tensor,
        value: Optional[torch.Tensor] = None,
        block_hashs: Optional[List[int]] = None,
        block_linear_idxs: Optional[List[int]] = None,
        len_to_big_page_id: Optional[SortedDict] = None,
    ) -> Tuple[int, Optional[LinearAttPagedTreeNode]]:
        assert key is not None
        if value is None:
            value = key
        assert len(key) == len(value)
        if block_hashs is None:
            block_hashs = []
        if block_linear_idxs is None:
            block_linear_idxs = []
        if len_to_big_page_id is None:
            len_to_big_page_id = SortedDict()

        assert (len(block_hashs) // self.big_page_num) >= len(len_to_big_page_id)

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

        ans = self._insert_helper(self.root_node, key, value, block_hashs, block_linear_idxs, len_to_big_page_id)
        assert len(len_to_big_page_id) == 0
        return ans

    def _insert_helper(
        self,
        node: LinearAttPagedTreeNode,
        key: torch.Tensor,
        value: torch.Tensor,
        block_hashs: List[int],
        block_linear_idxs: List[int],
        len_to_big_page_id: SortedDict,
    ) -> Tuple[int, Optional[LinearAttPagedTreeNode]]:
        self._discard_node(node)
        node.update_time()

        try:
            if len(block_hashs) == 0:
                return 0, node
            # 先看是不是能插入一个大页节点
            if len(block_hashs) >= self.big_page_num:
                # 插入大叶节点
                big_page_block_hash = block_hashs[self.big_page_num - 1]
                big_page_token_id_key = key[: self.big_page_tokens]
                big_page_token_mem_index_value = value[: self.big_page_tokens]
                if big_page_block_hash in node.children:
                    assert node.is_big_page_node()
                    child = node.children[big_page_block_hash]
                    assert child.is_big_page_node()

                    # 提前释放 len_to_big_page_id 对应的buffer资源
                    new_big_page_buffer_id = len_to_big_page_id.pop(child.node_prefix_total_len, None)
                    if new_big_page_buffer_id is not None:
                        # 因为节点已经存在，所以无法插入，但是要释放对应的buffer_id 节点
                        self.linear_att_big_page_buffers.free_state_cache([new_big_page_buffer_id])

                    # 已经存在了
                    sub_prefix_len, ans_node = self._insert_helper(
                        child,
                        key[self.big_page_tokens :],
                        value[self.big_page_tokens :],
                        block_hashs[self.big_page_num :],
                        block_linear_idxs[self.big_page_num :],
                        len_to_big_page_id,
                    )
                    return self.big_page_tokens + sub_prefix_len, ans_node
                else:
                    # 不存在，则新建一个大页节点
                    assert node.is_big_page_node()
                    new_big_page_buffer_id = len_to_big_page_id.pop(
                        node.node_prefix_total_len + self.big_page_tokens, None
                    )
                    assert new_big_page_buffer_id is not None

                    new_child = node.add_and_return_new_big_page_child(
                        big_page_token_id_key,
                        big_page_token_mem_index_value,
                        big_page_block_hash,
                        new_big_page_buffer_id,
                    )
                    self.tree_total_tokens_num.arr[0] += self.big_page_tokens
                    assert new_child.is_big_page_node()
                    assert new_child.page_num == self.big_page_num
                    _, ans_node = self._insert_helper(
                        new_child,
                        key[self.big_page_tokens :],
                        value[self.big_page_tokens :],
                        block_hashs[self.big_page_num :],
                        block_linear_idxs[self.big_page_num :],
                        len_to_big_page_id,
                    )
                    return 0, ans_node
            else:
                # 插入小页节点的情况
                assert len(block_hashs) < self.big_page_num

                # 是否已经存在了。
                if block_hashs[0] in node.children:
                    child = node.children[block_hashs[0]]

                    if block_linear_idxs[0] is not None:
                        assert len(block_hashs) == 1 == len(block_linear_idxs)
                        if child.small_page_buffer_idx is None:
                            # 将这个buffer id 移交给这个存在的节点。
                            self._discard_node(child)
                            child.small_page_buffer_idx = block_linear_idxs[0]
                            self._add_node(child)
                        else:
                            # 说明节点已经存在了，直接提前移除掉这个节点占用的线性缓存，外部不用处理这个细节了
                            self.linear_att_small_page_buffers.free_state_cache(free_indexes=[block_linear_idxs[0]])

                    sub_prefix_len, ans_node = self._insert_helper(
                        child,
                        key[self.hash_page_size :],
                        value[self.hash_page_size :],
                        block_hashs[1:],
                        block_linear_idxs[1:],
                        len_to_big_page_id,
                    )
                    return self.hash_page_size + sub_prefix_len, ans_node
                else:
                    new_node = node.add_and_return_new_child(
                        key[: self.hash_page_size],
                        value[: self.hash_page_size],
                        block_hashs[0],
                        block_linear_idxs[0],
                    )
                    assert not new_node.is_big_page_node()
                    assert new_node.page_num == 1
                    self.tree_total_tokens_num.arr[0] += self.hash_page_size
                    _, ans_node = self._insert_helper(
                        new_node,
                        key[self.hash_page_size :],
                        value[self.hash_page_size :],
                        block_hashs[1:],
                        block_linear_idxs[1:],
                        len_to_big_page_id,
                    )
                    return 0, ans_node

        finally:
            self._add_node(node)

    def match_prefix(
        self,
        key: torch.Tensor,
        block_hashs: Optional[List[int]] = None,
        update_refs: bool = False,
    ):
        assert update_refs is True, "update_refs must be True"
        assert key is not None, "key must not be None"
        if block_hashs is None:
            block_hashs = []

        assert (
            len(key) == len(block_hashs) * self.hash_page_size
        ), f"key length {len(key)} does not match block_hashs length {len(block_hashs)} * {self.hash_page_size}"

        if len(block_hashs) == 0 or len(key) == 0:
            return None, 0, None

        ans_node_list: List[LinearAttPagedTreeNode] = []
        self._match_prefix_helper(
            self.root_node,
            key=key,
            block_hashs=block_hashs,
            ans_node_list=ans_node_list,
            update_refs=update_refs,
        )
        if len(ans_node_list) == 0:
            return None, 0, None

        # 判定真正可以用的匹配节点。
        ans_node_list = self._trim_unusable_match_tail(ans_node_list)
        if len(ans_node_list) == 0:
            return None, 0, None

        ans_node = ans_node_list[-1]
        mem_value = torch.concat([e.token_mem_index_value for e in ans_node_list])
        assert len(mem_value) == ans_node.node_prefix_total_len

        return ans_node, len(mem_value), mem_value

    def _match_prefix_helper(
        self,
        node: LinearAttPagedTreeNode,
        key: torch.Tensor,
        block_hashs: Optional[List[int]],
        ans_node_list: list,
        update_refs: bool = False,
    ):
        self._discard_node(node)
        node.update_time()

        try:
            if update_refs:
                node.ref_counter += 1
                # from 0 to 1 need update refs token num
                if node.ref_counter == 1:
                    self.refed_tokens_num.arr[0] += len(node.token_mem_index_value)

            if len(block_hashs) == 0:
                return

            if len(block_hashs) >= self.big_page_num:
                # 大页的匹配
                big_page_block_hash = block_hashs[self.big_page_num - 1]
                if big_page_block_hash in node.children:
                    child = node.children[big_page_block_hash]
                    assert child.is_big_page_node()
                    ans_node_list.append(child)
                    self._match_prefix_helper(
                        child,
                        key[self.big_page_tokens :],
                        block_hashs[self.big_page_num :],
                        ans_node_list,
                        update_refs,
                    )
                    return

            # 小页匹配的情况
            if block_hashs[0] in node.children:
                child = node.children[block_hashs[0]]
                ans_node_list.append(child)
                self._match_prefix_helper(
                    child,
                    key[(self.hash_page_size) :],
                    block_hashs[1:],
                    ans_node_list,
                    update_refs,
                )
                return
            else:
                return

        finally:
            self._add_node(node)

    def _trim_unusable_match_tail(self, nodes: List[LinearAttPagedTreeNode]) -> List[LinearAttPagedTreeNode]:
        removed_list = []
        for node in reversed(nodes):
            if node.is_big_page_node():
                break
            elif node.small_page_buffer_idx is not None:
                assert not node.is_big_page_node()
                break
            else:
                removed_list.append(node)

        for node in removed_list:
            self._discard_node(node)
            # dec ref
            node.ref_counter -= 1
            if node.ref_counter == 0:
                self.refed_tokens_num.arr[0] -= len(node.token_mem_index_value)

            self._add_node(node)

        if len(removed_list) == 0:
            return nodes
        else:
            return nodes[: -len(removed_list)]

    def _try_merge(self, child_node: LinearAttPagedTreeNode) -> Optional[LinearAttPagedTreeNode]:
        raise NotImplementedError()

    def merge_unreferenced_nodes(self):
        raise NotImplementedError()

    def clear_tree_nodes(self):
        """Only used in tests."""
        self.free_radix_cache_to_get_enough_token(need_token_num=self.total_token_num)
        return

    def deref_to_first_big_page_node(self, node: LinearAttPagedTreeNode) -> Optional[LinearAttPagedTreeNode]:
        assert not node.is_big_page_node()
        iter_node = node
        while not iter_node.is_big_page_node():
            self._discard_node(iter_node)

            if iter_node.ref_counter == 1:
                self.refed_tokens_num.arr[0] -= len(iter_node.token_mem_index_value)
            iter_node.ref_counter -= 1

            self._add_node(iter_node)

            iter_node = iter_node.parent

        if iter_node is self.root_node:
            return None
        else:
            return iter_node

    def dec_node_ref_counter(self, node: LinearAttPagedTreeNode):
        if node is None:
            return
        old_node = node
        self._discard_node(old_node)

        while node is not None:
            if node.ref_counter == 1:
                self.refed_tokens_num.arr[0] -= len(node.token_mem_index_value)
            node.ref_counter -= 1
            node = node.parent

        self._add_node(old_node)
        return

    def add_node_ref_counter(self, node: LinearAttPagedTreeNode):
        if node is None:
            return
        old_node = node
        self._discard_node(old_node)

        while node is not None:
            if node.ref_counter == 0:
                self.refed_tokens_num.arr[0] += len(node.token_mem_index_value)
            node.ref_counter += 1
            node = node.parent

        self._add_node(old_node)
        return

    def get_mem_index_value_by_node(self, node: LinearAttPagedTreeNode) -> Optional[torch.Tensor]:
        if node is None:
            return None

        ans_list = []
        while node is not None:
            ans_list.append(node.token_mem_index_value)
            node = node.parent

        ans_list.reverse()
        return torch.concat(ans_list, dim=0)

    def get_big_page_ids_by_node(self, node: LinearAttPagedTreeNode) -> List[int]:
        if node is None:
            return []
        if node is self.root_node:
            return []

        ans_list = []
        while node is not self.root_node:
            if node.is_big_page_node():
                ans_list.append(node.big_page_buffer_idx)
            node = node.parent

        ans_list.reverse()
        return ans_list

    def get_refed_tokens_num(self):
        return self.refed_tokens_num.arr[0]

    def get_tree_total_tokens_num(self):
        return self.tree_total_tokens_num.arr[0]

    def print_self(self, indent=0):
        self._print_helper(self.root_node, indent)

    def _print_helper(self, node: LinearAttPagedTreeNode, indent):
        print(
            " " * indent,
            f"hash_info: {node.page_hash} "
            f"k: {node.token_id_key[0:10] if node.token_id_key is not None else None} "
            f"v: {node.token_mem_index_value[0:10] if node.token_mem_index_value is not None else None} "
            f"refs: {node.ref_counter} time_id: {node.time_id} "
            f"prefix_total_len: {node.node_prefix_total_len} "
            f"node_value_len: {node.node_value_len} buffer_idx: {node.small_page_buffer_idx}",
        )
        for _, child in node.children.items():
            self._print_helper(child, indent=indent + 2)
        return

    def free_radix_cache_to_get_enough_token(self, need_token_num):
        assert self.mem_manager is not None
        if need_token_num > self.mem_manager.can_use_mem_size:
            need_evict_token_num = need_token_num - self.mem_manager.can_use_mem_size
            release_mems = []
            small_page_buffer_ids = []

            def release_mem(mem_index, linear_att_small_page_id):
                release_mems.append(mem_index)
                small_page_buffer_ids.append(linear_att_small_page_id)
                return

            self._evict(need_evict_token_num, release_mem)
            mem_index = torch.concat(release_mems)
            self.mem_manager.free(mem_index)
            small_page_buffer_ids = [idx for idx in small_page_buffer_ids if idx is not None]
            if len(small_page_buffer_ids) > 0:
                self.linear_att_small_page_buffers.free_state_cache(small_page_buffer_ids)
        return

    def free_one_small_page_linear_att_buffer(self):
        if self.linear_att_small_page_buffers is None:
            return
        if self.linear_att_small_page_buffers.get_free_cache_num() > 0:
            return
        if len(self._evict_tree_set_for_linear_att) == 0:
            return

        node: LinearAttPagedTreeNode = self._evict_tree_set_for_linear_att.pop(0)
        self._discard_node(node)

        assert node.small_page_buffer_idx is not None
        self.linear_att_small_page_buffers.free_state_cache(free_indexes=[node.small_page_buffer_idx])
        node.small_page_buffer_idx = None

        self._add_node(node)
        return

    def _evict(self, need_remove_tokens, evict_callback):
        if self.tree_total_tokens_num.arr[0] - self.refed_tokens_num.arr[0] < need_remove_tokens:
            assert False, f"""can not free tree tokens {need_remove_tokens},
                              tree_total_tokens_num {self.tree_total_tokens_num.arr[0]},
                              refed_tokens_num {self.refed_tokens_num.arr[0]}"""
        num_evicted = 0
        while num_evicted < need_remove_tokens:
            node: LinearAttPagedTreeNode = self._evict_tree_set.pop(0)
            self._discard_node(node)

            assert (
                node.ref_counter == 0 and len(node.children) == 0 and node is not self.root_node
            ), "error evict tree node state"
            num_evicted += len(node.token_mem_index_value)

            if node.is_big_page_node():
                assert node.big_page_buffer_idx is not None
                self.linear_att_big_page_buffers.free_state_cache([node.big_page_buffer_idx])

            evict_callback(node.token_mem_index_value, node.small_page_buffer_idx)
            self.tree_total_tokens_num.arr[0] -= len(node.token_mem_index_value)
            parent_node: LinearAttPagedTreeNode = node.parent
            parent_node.remove_child(node)

            self._add_node(parent_node)

        return
