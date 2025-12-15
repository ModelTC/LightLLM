from typing import Set, Protocol, List, Optional, Tuple

import torch
from sortedcontainers import SortedSet

from lightllm.server.router.dynamic_prompt.radix_cache import RadixCache, TreeNode
from lightllm.common.kv_cache_mem_manager.mem_manager import MemoryManager


class HybridMemManager(MemoryManager):
    def alloc_buffer(self, need_size):
        ...

    def free_buffer(self, free_buffer_indexes):
        ...

    def get_buffer(self, layer_index):
        ...

    def get_buffer_can_use_size(self):
        ...

    def copy_buffer(self, src_idx, tgt_idx):
        ...


class HybridRadixCache(RadixCache):
    def __init__(self, unique_name, total_token_num, rank_in_node, mem_manager=None):
        self.mem_manager: HybridMemManager = mem_manager
        super().__init__(unique_name, total_token_num, rank_in_node, mem_manager)
        # 用于缓存需要被驱逐的buffer节点， 应该包含所有有buffer的节点
        self.evict_buffer_set: Set[TreeNode] = SortedSet(key=lambda x: (x.buffer_time,))

    def free_radix_cache_to_get_enough_buffer(self, need_buffer_num):
        if need_buffer_num > self.mem_manager.get_buffer_can_use_size():
            need_evict_buffer_num = need_buffer_num - self.mem_manager.get_buffer_can_use_size()

            release_mems = []

            def release_mem(mem_index):
                release_mems.append(mem_index)
                return

            release_buffers = []

            def release_buffer(buffer_idx):
                release_buffers.append(buffer_idx)
                return

            self.evict_buffer(need_evict_buffer_num, release_buffer, release_mem)
            self.mem_manager.free_buffer(release_buffers)
            if len(release_mems) > 0:
                mem_index = torch.concat(release_mems)
                self.mem_manager.free(mem_index)
        return

    def evict_buffer(self, need_evict_buffer_num, evict_buffer_callback, evict_token_callback):
        while need_evict_buffer_num > 0:
            node = self.evict_buffer_set.pop(0)
            assert node.buffer_idx is not None
            evict_buffer_callback(node.buffer_idx)
            need_evict_buffer_num -= 1
            # 当一个节点的buffer_idx变为None时，事实上无法在后续进行match，
            # 但当该节点子节点或者引用数不为0时，仍然需要保留， 否则则应该被删除
            if node.is_leaf() and node.ref_counter == 0:
                self.evict_tree_set.discard(node)
                evict_token_callback(node.token_mem_index_value)
                self.tree_total_tokens_num.arr[0] -= len(node.token_mem_index_value)
                parent_node: TreeNode = node.parent
                parent_node.remove_child(node)
                if parent_node.is_leaf():
                    self.evict_tree_set.add(parent_node)
        return

    def insert_for_hybrid_radix_cache(self, reqs):
        # 在请求运行途中对prefix cache进行保留，而不是请求被释放时
        from lightllm.server.router.model_infer.infer_batch import g_infer_context
        from lightllm.common.basemodel.infer_lock import g_infer_state_lock

        # 过滤掉 cur_kv_len 为 0 的请求（新请求还没有生成任何 KV）
        valid_reqs = [req for req in reqs if req.cur_kv_len > 0]

        if len(valid_reqs) == 0:
            return

        # 确保有足够的空间用于新的 buffer，并在锁保护下完成所有 radix cache 操作
        g_infer_state_lock.acquire()
        try:
            self.free_radix_cache_to_get_enough_buffer(len(valid_reqs))
            new_buffer_indexes = self.mem_manager.alloc_buffer(len(valid_reqs))

            for i, req in enumerate(valid_reqs):
                input_token_ids = req.get_input_token_ids()
                key = torch.tensor(input_token_ids[0 : req.cur_kv_len], dtype=torch.int64, device="cpu")
                value = g_infer_context.req_manager.req_to_token_indexs[req.req_idx][: req.cur_kv_len].cpu()
                cur_buffer_idx = g_infer_context.req_manager.req_to_buffer_indexes[req.req_idx]

                # 分配新的 buffer 并复制当前 buffer 的内容
                self.mem_manager.copy_buffer(cur_buffer_idx, new_buffer_indexes[i])

                _, new_shared_kv_node = super().insert(key, value)
                self.dec_node_ref_counter(req.shared_kv_node)
                self.add_node_ref_counter(new_shared_kv_node)
                new_shared_kv_node.buffer_idx = new_buffer_indexes[i]
                req.shared_kv_node = new_shared_kv_node
        finally:
            g_infer_state_lock.release()

    def match_prefix(self, key, update_refs=False):
        assert len(key) != 0
        ans_value_list = []
        tree_node = self._match_prefix_helper(self.root_node, key, ans_value_list, update_refs=update_refs)
        evict_token_list = []
        while tree_node != self.root_node and tree_node.buffer_idx is None:
            self.dec_node_ref_counter(tree_node)
            if tree_node.is_leaf() and tree_node.ref_counter == 0:
                self.evict_tree_set.discard(tree_node)
                evict_token_list.append(tree_node.token_mem_index_value)
                self.tree_total_tokens_num.arr[0] -= len(tree_node.token_mem_index_value)
                parent_node: TreeNode = tree_node.parent
                parent_node.remove_child(tree_node)
                if parent_node.is_leaf():
                    self.evict_tree_set.add(parent_node)
                tree_node = parent_node
            else:
                tree_node = tree_node.parent
            ans_value_list.pop()

        if len(evict_token_list) > 0:
            evict_token_value = torch.concat(evict_token_list)
            self.mem_manager.free(evict_token_value)

        if tree_node == self.root_node:
            return None, 0, None

        update_node = tree_node
        while update_node != self.root_node:
            if update_node.buffer_idx is not None:
                self.evict_buffer_set.discard(update_node)
                update_node.update_buffer_time()
                self.evict_buffer_set.add(update_node)
            update_node = update_node.parent

        value = torch.concat(ans_value_list)
        return tree_node, len(value), value

    def insert(self, key, value) -> Tuple[int, Optional[TreeNode]]:
        prefix_len, node = super().insert(key, value)
        self.evict_buffer_set.discard(node)
        node.update_buffer_time()
        self.evict_buffer_set.add(node)
        return prefix_len, node

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
                self.mem_manager.free_buffer(release_buffers)
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
            ), "error evict tree node state"
            num_evicted += len(node.token_mem_index_value)
            evict_callback(node.token_mem_index_value)
            if node.buffer_idx is not None:
                self.evict_buffer_set.discard(node)
                evict_buffer_callback(node.buffer_idx)
            # update total token num
            self.tree_total_tokens_num.arr[0] -= len(node.token_mem_index_value)
            parent_node: TreeNode = node.parent
            parent_node.remove_child(node)
            if parent_node.is_leaf():
                self.evict_tree_set.add(parent_node)

        return
