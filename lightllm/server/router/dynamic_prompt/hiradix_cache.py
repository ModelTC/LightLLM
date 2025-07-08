import torch
import time
import tempfile
import numpy as np
import torch.distributed as dist
from os.path import join
from .radix_cache import RadixCache, TreeNode, match
from typing import Tuple, Dict, Set, List
from lightllm.common.mem_manager import MemoryManager
from lightllm.utils.log_utils import init_logger
from threading import Lock
from enum import Enum
from .shared_arr import SharedArray
from kvcache.python.jit import PyLocalCacheService

logger = init_logger(__name__)

def wait_until_ready(task, timeout=10.0, check_interval=0.01):
    start_time = time.time()
    while not task.ready():
        time.sleep(check_interval)
        if time.time() - start_time > timeout:
            logger.error("Current kv cache task not ready in time")
            return False
    return True

class LocalCacheManager:

    def __init__(self, unique_name: str, rank_in_node: int, mem_manager):
        tmp_dir = tempfile.mkdtemp(prefix=f"cache_{unique_name}_{rank_in_node}")
        self.cache_file = join(tmp_dir, "cache_file")
        all_buffers = mem_manager.kv_buffer
        all_buffers = all_buffers.view(all_buffers.shape[0], all_buffers.shape[1], -1)

        self.py_cache_service = PyLocalCacheService(
            file=self.cache_file,
            storage_size=128 * (1024 ** 3),  # 128GB
            num_shard=32,
            kvcache_tensor=all_buffers,
            num_worker=8
        )

    def insert(self, tokens, kv_page_indexer, start_pos=0):
        t = self.py_cache_service.create(
                tokens=tokens, 
                kv_page_indexer=kv_page_indexer, 
                mode="w",
                start_pos=start_pos)
        res = wait_until_ready(t)
        if not res:
            self.py_cache_service.az5(t)

    def read(self, tokens, kv_page_indexer, start_pos=0):
        t = self.py_cache_service.create(
                tokens=tokens, 
                kv_page_indexer=kv_page_indexer, 
                mode="r",
                start_pos=start_pos)
        res = wait_until_ready(t)
        return res

    def query(self, tokens):
        query_result = self.py_cache_service.query(tokens)
        max_len = 0
        for result in query_result:
            if result:
                max_len += 1
            else:
                break
        return max_len * self.block_size

    @property
    def block_size(self,):
        return self.py_cache_service.tokens_per_block

class HiRadixCache(RadixCache):
    def __init__(self, unique_name, total_token_num, rank_in_node, mem_manager):
        super().__init__(unique_name, total_token_num, rank_in_node, mem_manager)
        self.rank_in_node = rank_in_node
        self.local_cache_manager = LocalCacheManager(
            unique_name,
            rank_in_node,
            mem_manager,
        )
        self.is_hi_radix_cache = True
        self.disk_cache_match_count = SharedArray(f"{unique_name}_disk_cache_match_count_{rank_in_node}", (1,), dtype=np.int64)
        self.disk_cache_match_count.arr[0] = 0
        self.total_match_count = SharedArray(f"{unique_name}_total_match_count_{rank_in_node}", (1,), dtype=np.int64)
        self.total_match_count.arr[0] = 0
        self.disk_cache_match_ratio = SharedArray(f"{unique_name}_disk_cache_match_ratio_{rank_in_node}", (1,), dtype=np.float32)
        self.disk_cache_match_ratio.arr[0] = 0.0
        logger.info(f"Initializing HiRadixCache {rank_in_node}")

    def insert(self, key, value=None):
        share_len = super().insert(key, value)
        if share_len == 0:
            return 0
        self.local_cache_manager.insert(key, value)
        return share_len

    def match_prefix(self, key, update_refs=False):
        assert len(key) != 0
        self.total_match_count.arr[0] += 1
        ans_value_list = []
        ans_value = None
        tree_node = self._match_prefix_helper(self.root_node, key, ans_value_list, update_refs=False)
        if tree_node.node_prefix_total_len != 0:
            ans_value = torch.concat(ans_value_list)
        max_len = 0
        if tree_node.node_prefix_total_len < len(key):
            max_len = self.local_cache_manager.query(key)
        if max_len > tree_node.node_prefix_total_len:
            pull_len = max_len - tree_node.node_prefix_total_len
            self.disk_cache_match_count.arr[0] += 1
            self.disk_cache_match_ratio.arr[0] = self.disk_cache_match_count.arr[0] / self.total_match_count.arr[0]
            self.free_radix_cache_to_get_enough_token(pull_len)
            buffers = self.mem_manager.alloc(pull_len)
            start_pos = 0
            if ans_value is not None:
                buffers = torch.concat([ans_value, buffers])
                start_pos = (tree_node.node_prefix_total_len - 1) // self.local_cache_manager.block_size * self.local_cache_manager.block_size
            logger.debug(f"HiCache current match ratio {self.disk_cache_match_ratio.arr[0]}, pulled cache len {pull_len} from disk")
            res = self.local_cache_manager.read(tokens=key[:max_len], kv_page_indexer=buffers, start_pos=start_pos)
            if res:
                super().insert(key[:max_len], buffers)
            else:
                self.mem_manager.free(buffers[tree_node.node_prefix_total_len:])
        return super().match_prefix(key, update_refs=update_refs)
