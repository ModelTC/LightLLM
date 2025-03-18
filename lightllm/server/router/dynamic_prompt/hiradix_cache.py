import torch
from .cache_controller import HiCacheController
from .radix_cache import RadixCache, TreeNode, match
from typing import Tuple, Dict, Set, List
from lightllm.common.mem_manager import MemoryManager


class HiRadixCache(RadixCache):
    def __init__(self, cache_controller: HiCacheController, unique_name, total_token_num, rank_in_node, mem_manager: MemoryManager = None):
        super().__init__(unique_name, total_token_num, rank_in_node, mem_manager)
        self.cache_controller = cache_controller
    
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
