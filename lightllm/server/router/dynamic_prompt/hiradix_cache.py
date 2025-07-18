import torch
import time
import tempfile
import zmq
import inspect
import threading
import numpy as np
import torch.multiprocessing as mp
import torch.distributed as dist
from os.path import join
from .radix_cache import RadixCache, TreeNode, match
from typing import Tuple, Dict, Set, List
from lightllm.common.mem_manager import MemoryManager
from lightllm.common.radixmem_buffer import RadixMemoryBuffer
from lightllm.common.radixmem_manager import RadixBufferManager
from lightllm.utils.log_utils import init_logger
from threading import Lock
from enum import Enum
from .shared_arr import SharedArray
from .io_objs import ShmReqInfo
from lightllm.server.core.objs import Req, RadixStatus
from lightllm.server.core.objs.io_objs import GroupReqIndexes
from lightllm.server.core.objs import ShmReqManager
from lightllm.utils.graceful_utils import graceful_registry
from lightllm.server.core.objs import Req, SamplingParams, FinishStatus, ShmReqManager

logger = init_logger(__name__)


class LocalCacheManager:

    def __init__(self, radix_manager: RadixBufferManager, mem_manager: MemoryManager, rank_in_node):
        self.radix_manager = radix_manager
        self.radix_buffer: RadixMemoryBuffer = self.radix_manager.radix_buffer
        self.mem_manager = mem_manager
        self.rank_in_node = rank_in_node

    def insert(self, req: Req, key: torch.Tensor, value=None):
        query_len, query_index = self._query_cache(req, key)

        alloc_len = len(key) - query_len
        if alloc_len == 0:
            self._set_radix_staus(req, RadixStatus.WRITE_READY)
            return

        new_index = self._alloc_and_copy_kv(alloc_len, value)

        start_pos = max(0, (query_len - 1) // self.chunk_size * self.chunk_size)
        self.radix_manager.write(
            tokens=key.tolist(),
            values=query_index + new_index,
            start_pos=start_pos
        )

        self._set_radix_staus(req, RadixStatus.WRITE_READY)
    
    def _query_cache(self, req, key):
        if req.radix_status.is_no_need_cache(self.rank_in_node):
            logger.info(f"query no need cache {self.rank_in_node} {req.radix_status.get_status(self.rank_in_node)}")
            return 0, []
        
        if req.radix_status.is_read_ready(self.rank_in_node):
            query_len, mem_index = self.radix_manager.query_cache(key.tolist())
            return query_len, mem_index
        return 0, []

    def _alloc_and_copy_kv(self, alloc_len, value):
        assert alloc_len > 0, "No allocation needed"

        new_index = self.radix_buffer.alloc(alloc_len)
        dst_kv_buffer = self.radix_buffer.get_kv_buffer(new_index)
        src_kv_buffer = self.mem_manager.get_index_kv_buffer(value[-alloc_len:])["kv_buffer"]

        assert len(src_kv_buffer) == len(dst_kv_buffer), f"Mis match buffer size src {len(src_kv_buffer)} != dst {len(dst_kv_buffer)}"

        self.copy_kv_from_gpu_to_cpu(src_kv_buffer, dst_kv_buffer)
        return new_index.tolist()

    def _set_radix_staus(self, req, status):
        req.radix_status.set_status(self.rank_in_node, status)

    def read(self, key, value, query_index, alloc_len):
        try:
            src_kv_buffer = self.radix_buffer.get_kv_buffer(index=query_index[-alloc_len:])
            dst_kv_buffer = self.mem_manager.get_index_kv_buffer(index=value[-alloc_len:])["kv_buffer"]

            assert len(src_kv_buffer) == len(dst_kv_buffer), f"Mis match buffer size src {len(src_kv_buffer)} != dst {len(dst_kv_buffer)}"

            self.copy_kv_from_cpu_to_gpu(src_kv_buffer, dst_kv_buffer)

        except Exception as e:
            logger.error(f"LocalCache read from radix mem error {e}")
            return False

        return True

    def query(self, req: Req, key):
        return self._query_cache(req, key)
    
    @property
    def chunk_size(self):
        return self.radix_manager.chunk_size
    
    def copy_kv_from_cpu_to_gpu(self, src_kv_tensor, dst_kv_tensor):
        dst_kv_tensor.copy_(src_kv_tensor, non_blocking=True)
        
    def copy_kv_from_gpu_to_cpu(self, src_kv_tensor, dst_kv_tensor):
        dst_kv_tensor.copy_(src_kv_tensor, non_blocking=True)  


class HiRadixCache(RadixCache):
    def __init__(self, unique_name, total_token_num, rank_in_node, mem_manager, radix_manager, radix_info_queue):
        super().__init__(unique_name, total_token_num, rank_in_node, mem_manager)
        self.rank_in_node = rank_in_node
        self.radix_manager: RadixBufferManager = radix_manager
        self.local_cache_manager = LocalCacheManager(
            radix_manager=self.radix_manager,
            mem_manager=mem_manager,
            rank_in_node=rank_in_node
        )
        self.radix_info_queue = radix_info_queue
        self.is_hi_radix_cache = True
        self.disk_cache_match_count = SharedArray(f"{unique_name}_disk_cache_match_count_{rank_in_node}", (1,), dtype=np.int64)
        self.disk_cache_match_count.arr[0] = 0
        self.total_match_count = SharedArray(f"{unique_name}_total_match_count_{rank_in_node}", (1,), dtype=np.int64)
        self.total_match_count.arr[0] = 0
        self.disk_cache_match_ratio = SharedArray(f"{unique_name}_disk_cache_match_ratio_{rank_in_node}", (1,), dtype=np.float32)
        self.disk_cache_match_ratio.arr[0] = 0.0
        logger.info(f"Initializing HiRadixCache {rank_in_node}")

    def insert(self, key, value=None, req=None):
        if len(key) == 0:
            return 0
        share_len = super().insert(key, value)
        if req is None:
            return
        self.local_cache_manager.insert(req, key, value)
        return share_len

    def match_prefix(self, req, key, update_refs=False):
        assert len(key) != 0
        self.total_match_count.arr[0] += 1
        ans_value_list = []
        ans_value = None
        tree_node = self._match_prefix_helper(self.root_node, key, ans_value_list, update_refs=False)
        if tree_node.node_prefix_total_len != 0:
            ans_value = torch.concat(ans_value_list)
        max_len = 0
        if tree_node.node_prefix_total_len < len(key):
            max_len, query_index = self.local_cache_manager.query(req, key)

        logger.debug(f"HiCache rank_in_node={self.rank_in_node} current key len {len(key)} match radix len {tree_node.node_prefix_total_len}, max len {max_len}")
        if max_len > tree_node.node_prefix_total_len:
            pull_len = max_len - tree_node.node_prefix_total_len
            self.disk_cache_match_count.arr[0] += 1
            self.disk_cache_match_ratio.arr[0] = self.disk_cache_match_count.arr[0] / self.total_match_count.arr[0]
            self.free_radix_cache_to_get_enough_token(pull_len)
            buffers = self.mem_manager.alloc(pull_len)
            if ans_value is not None:
                buffers = torch.concat([ans_value, buffers])
            logger.debug(f"HiCache current match ratio {self.disk_cache_match_ratio.arr[0]}, pulled cache len {pull_len} from disk")
            res = self.local_cache_manager.read(key[:max_len], buffers, query_index, alloc_len=pull_len)
            if res:
                super().insert(key[:max_len], buffers)
            else:
                self.mem_manager.free(buffers[tree_node.node_prefix_total_len:])
            
        return super().match_prefix(key, update_refs=update_refs)
