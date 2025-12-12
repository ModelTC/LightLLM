from typing import Set, Protocol, List

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
        self.evict_buffer_set: Set[TreeNode] = SortedSet(key=lambda x: x.time_id)
        self.evict_buffer_set.add(self.root_node)

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
            node = self.evict_buffer_set.pop()
            if node.buffer_idx is not None:
                evict_buffer_callback(node.buffer_idx)
                need_evict_buffer_num -= 1
            else:
                # 在混合注意力模型的情景里，只能匹配 buffer_idx 不为 None的节点
                # 假如 buffer_idx 为 None，则当做匹配失败。
                # 所以可以直接把这个节点给释放掉
                if node.is_leaf() and node.ref_counter == 0:
                    self._remove_leaf_node(node)
        return

    def insert_for_hybrid_radix_cache(self, reqs):
        from lightllm.server.router.model_infer.infer_batch import g_infer_context
        from lightllm.common.basemodel.infer_lock import g_infer_state_lock

        # 确保有足够的空间用于新的 buffer
        g_infer_state_lock.acquire()
        self.free_radix_cache_to_get_enough_buffer(len(reqs))
        new_buffer_indexes = self.mem_manager.alloc_buffer(len(reqs))
        g_infer_state_lock.release()

        for i, req in enumerate(reqs):
            input_token_ids = req.get_input_token_ids()
            key = torch.tensor(input_token_ids[0 : req.cur_kv_len], dtype=torch.int64, device="cpu")
            value = g_infer_context.req_manager.req_to_token_indexs[req.req_idx][: req.cur_kv_len].cpu()
            cur_buffer_idx = g_infer_context.req_manager.req_to_buffer_indexes[req.req_idx]
            # 分配新的 buffer 并复制当前 buffer 的内容
            self.mem_manager.copy_buffer(cur_buffer_idx, new_buffer_indexes[i])

            _, new_shared_kv_node = self.insert(key, value)
            new_shared_kv_node.buffer_idx = new_buffer_indexes[i]
            self.dec_node_ref_counter(req.shared_kv_node)
            self.add_node_ref_counter(new_shared_kv_node)
            req.shared_kv_node = new_shared_kv_node

    def match_prefix(self, key, update_refs=False):
        assert len(key) != 0
        ans_value_list = []
        tree_node = self._match_prefix_helper(self.root_node, key, ans_value_list, update_refs=update_refs)

        while tree_node != self.root_node and tree_node.buffer_idx is None:
            self.dec_node_ref_counter(tree_node)
            if tree_node.is_leaf() and tree_node.ref_counter == 0:
                tree_node = self._remove_leaf_node(tree_node)
            else:
                tree_node = tree_node.parent
            ans_value_list.pop()

        if tree_node == self.root_node:
            return None, 0, None

        value = torch.concat(ans_value_list)
        return tree_node, len(value), value

    def _remove_leaf_node(self, node: TreeNode):
        self.evict_tree_set.discard(node)
        self.tree_total_tokens_num.arr[0] -= len(node.token_mem_index_value)
        parent_node: TreeNode = node.parent
        parent_node.remove_child(node)
        if parent_node.is_leaf():
            self.evict_tree_set.add(parent_node)
        return parent_node
