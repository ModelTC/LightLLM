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

    def __init__(self, mem_buffer: RadixMemoryBuffer, mem_manager: MemoryManager, rank_in_node):
        self.mem_buffer = mem_buffer
        self.mem_manager = mem_manager
        self.rank_in_node = rank_in_node

    def insert(self, req: Req, key: torch.Tensor, value=None):
        pre_index = self.mem_buffer.get_req_mem_index(req.request_id)
        if len(pre_index) != 0 and len(value) > len(pre_index):
            logger.info(f"pre index req {req.request_id} {pre_index}")
            alloc_len = len(value) - len(pre_index)
            index = self.mem_buffer.alloc(alloc_len)
            value = value[len(pre_index):]
            self.mem_buffer.set_req_mem_index(
                req.request_id, pre_index + index.tolist()
            )
            logger.info(f"udpate index req {req.request_id} {pre_index + index.tolist()}")
        else:
            index = self.mem_buffer.alloc(len(value))
            self.mem_buffer.set_req_mem_index(
                req.request_id, index.tolist()
            )
        dst_kv_buffer = self.mem_buffer.get_kv_buffer(index)
        src_kv_buffer = self.mem_manager.get_index_kv_buffer(value)["kv_buffer"]
        logger.info(f"insert mem_buffer shape {dst_kv_buffer.shape}, manager buffer shape {src_kv_buffer.shape}")
        assert len(src_kv_buffer) == len(dst_kv_buffer), f"src kv buffer len {len(src_kv_buffer)} != dst kv buffer len {len(dst_kv_buffer)}"
        self.copy_kv_from_gpu_to_cpu(src_kv_buffer, dst_kv_buffer)
        req.radix_status.set_status(self.rank_in_node, RadixStatus.WRITE_READY)

    def read(self, req: Req, dst_index):
        try:
            index = self.mem_buffer.get_req_mem_index(req.group_req_id)
            src_kv_buffer = self.mem_buffer.get_kv_buffer(index[-len(dst_index)])
            dst_kv_buffer = self.mem_manager.get_index_kv_buffer(dst_index)["kv_buffer"]
            logger.info(f"len mem src index and dst index {len(index), len(dst_index)} read mem_buffer shape {src_kv_buffer.shape}, manager buffer shape {dst_kv_buffer.shape}")
            assert len(src_kv_buffer) == len(dst_kv_buffer), f"src kv buffer len {len(src_kv_buffer)} != dst kv buffer len {len(dst_kv_buffer)}"
            self.copy_kv_from_cpu_to_gpu(src_kv_buffer, dst_kv_buffer)
            #TODO no free
            self.mem_buffer.free_req_index(req.group_req_id)
        except Exception as e:
            logger.error(f"Local cache read from radix mem_buffer error {e}")
            return False
        return True

    def query(self, req: Req):
        if req.radix_status.is_no_need_cache(self.rank_in_node):
            logger.info(f"query no need cache {self.rank_in_node} {req.radix_status.get_status(self.rank_in_node)}")
            return 0
        if req.radix_status.is_read_ready(self.rank_in_node):
            index = self.mem_buffer.get_req_mem_index(req.group_req_id)
            logger.info(f"query find cache {self.rank_in_node} {req.radix_status.get_status(self.rank_in_node)} len {len(index)}")
            return len(index)
        return 0
    
    def copy_kv_from_cpu_to_gpu(self, src_kv_tensor, dst_kv_tensor):
        dst_kv_tensor.copy_(src_kv_tensor, non_blocking=True)
        
    def copy_kv_from_gpu_to_cpu(self, src_kv_tensor, dst_kv_tensor):
        dst_kv_tensor.copy_(src_kv_tensor, non_blocking=True)  


class HiRadixCache(RadixCache):
    def __init__(self, unique_name, total_token_num, rank_in_node, mem_manager, mem_buffer, radix_info_queue):
        super().__init__(unique_name, total_token_num, rank_in_node, mem_manager)
        self.rank_in_node = rank_in_node
        self.local_cache_manager = LocalCacheManager(
            mem_buffer=mem_buffer,
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
            max_len = self.local_cache_manager.query(req)
        logger.debug(f"HiCache rank_in_node={self.rank_in_node} current key len {len(key)} match radix len {tree_node.node_prefix_total_len}, max len {max_len}")
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
            # res = self.local_cache_manager.read(tokens=key[:max_len], kv_page_indexer=buffers, start_pos=start_pos)
            res = self.local_cache_manager.read(req, buffers)
            if res:
                super().insert(key[:max_len], buffers)
            else:
                self.mem_manager.free(buffers[tree_node.node_prefix_total_len:])
            
        return super().match_prefix(key, update_refs=update_refs)
