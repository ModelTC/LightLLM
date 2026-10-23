from collections import deque
from typing import List

import torch

from lightllm.common.kv_cache_mem_manager import MemoryManager
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.log_utils import init_logger


logger = init_logger("lightllm.common.req_manager")


class _ReqIndexPool:
    def __init__(self, max_request_num):
        self._free_indexes = deque(range(max_request_num))
        self.marks = [0 for _ in range(max_request_num)]
        self.can_alloc_size = max_request_num
        return

    def alloc(self):
        if not self._free_indexes:
            logger.warning("alloc req index fail")
            return None
        index = self._free_indexes.popleft()
        assert self.marks[index] == 0
        self.marks[index] = 1
        self.can_alloc_size -= 1
        return index

    def free(self, index):
        assert self.marks[index] == 1
        # 保持原有头插行为，优先复用最近释放的请求索引。
        self._free_indexes.appendleft(index)
        self.marks[index] = 0
        self.can_alloc_size += 1
        return

    def is_all_free(self):
        return self.can_alloc_size == len(self.marks)


class ReqManager:
    def __init__(self, max_request_num, max_sequence_length, mem_manager: MemoryManager):
        from .req_sampling_params import ReqSamplingParamsManager

        # 这里对最大请求数量的管理在默认上多申请了一个，主要是 index 为 max_request_num 代表
        # 的这个请求管理 id， 主要是为了兼容 DP 运行模式下，让各个 DP 能 padding 到 DP 中最大
        # 的那个batch size 进行运行，所有 padding 的请求都会使用预留的这个请求管理 id 进行处理
        # 这样让 DP 的实现更为简化一些。
        self.req_list = _ReqIndexPool(max_request_num)
        page_size = get_env_start_args().page_size
        self.page_size = page_size
        aligned_sequence_length = (max_sequence_length + page_size - 1) // page_size * page_size
        self.req_to_token_indexs = torch.zeros(
            (max_request_num + 1, aligned_sequence_length), dtype=torch.int32, device="cuda"
        )
        self.req_sampling_params_manager = ReqSamplingParamsManager(max_request_num)
        self.max_request_num = max_request_num
        self.HOLD_REQUEST_ID = max_request_num
        self.mem_manager = None
        if mem_manager is not None:
            self.bind_mem_manager(mem_manager)

    def bind_mem_manager(self, mem_manager: MemoryManager):
        self.mem_manager = mem_manager

        self.init_hold_request_indexs()
        return

    def init_hold_request_indexs(self):
        assert (
            self.req_list.is_all_free()
        ), "hold request indexes can only be initialized when all requests are released"

        # HOLD_REQUEST_ID 对应的请求行供 DP padding、overlap microbatch 等占位请求使用。将该行
        # 按 page_size 划分后，每一页都映射到 mem_manager 额外保留的同一个物理页；这样占位请求
        # 无论访问哪一个逻辑位置，都会落到合法且不会参与正常分配的 KV cache 地址上。
        assert self.mem_manager.page_size == self.page_size
        hold_row = self.req_to_token_indexs[self.HOLD_REQUEST_ID]
        token_offsets = torch.arange(
            hold_row.numel(), dtype=hold_row.dtype, device=hold_row.device, requires_grad=False
        )
        hold_page_start = self.mem_manager.HOLD_TOKEN_MEMINDEXES[0]
        hold_row.copy_(token_offsets.remainder(self.page_size).add_(hold_page_start))
        return

    def alloc(self):
        return self.req_list.alloc()

    def free(self, free_req_indexes: List[int], free_token_index):
        for req_index in free_req_indexes:
            self.req_list.free(req_index)

        if self.req_list.is_all_free():
            logger.debug(f"freed all request size {self.req_list.can_alloc_size}")
        self.mem_manager.free(free_token_index)

    def free_req(self, free_req_index: int):
        self.req_list.free(free_req_index)
        if self.req_list.is_all_free():
            logger.debug(f"freed all request size {self.req_list.can_alloc_size}")
        return

    def free_token(self, free_token_index):
        self.mem_manager.free(free_token_index)
        return

    def free_all(self):
        self.req_list = _ReqIndexPool(self.max_request_num)
        return
