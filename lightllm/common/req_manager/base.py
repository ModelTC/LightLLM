from typing import List

import torch

from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.log_utils import init_logger


logger = init_logger("lightllm.common.req_manager")


class _ReqIndexPool:
    def __init__(self, max_request_num):
        self.free_indexes = list(range(max_request_num - 1, -1, -1))
        self.marks = [0] * max_request_num

    @property
    def can_alloc_size(self):
        return len(self.free_indexes)

    def alloc(self):
        if not self.free_indexes:
            logger.warning("alloc req index fail")
            return None
        index = self.free_indexes.pop()
        assert self.marks[index] == 0
        self.marks[index] = 1
        return index

    def free(self, index):
        assert self.marks[index] == 1
        self.marks[index] = 0
        self.free_indexes.append(index)

    def is_all_free(self):
        return len(self.free_indexes) == len(self.marks)


class ReqManager:
    def __init__(self, max_request_num, max_sequence_length):
        from .req_sampling_params import ReqSamplingParamsManager

        # 这里对最大请求数量的管理在默认上多申请了一个，主要是 index 为 max_request_num 代表
        # 的这个请求管理 id， 主要是为了兼容 DP 运行模式下，让各个 DP 能 padding 到 DP 中最大
        # 的那个batch size 进行运行，所有 padding 的请求都会使用预留的这个请求管理 id 进行处理
        # 这样让 DP 的实现更为简化一些。
        self.req_list = _ReqIndexPool(max_request_num)
        self.page_size = page_size = get_env_start_args().page_size
        max_sequence_length = (max_sequence_length + page_size - 1) // page_size * page_size
        self.req_to_token_indexs = torch.zeros(
            (max_request_num + 1, max_sequence_length), dtype=torch.int32, device="cuda"
        )
        self.req_sampling_params_manager = ReqSamplingParamsManager(max_request_num)
        self.max_request_num = max_request_num
        self.HOLD_REQUEST_ID = max_request_num

    def bind_hold_page(self, token_indices):
        """Map padding requests to the resource owner's reserved physical page."""
        assert (
            self.req_list.is_all_free()
        ), "hold request indexes can only be initialized when all requests are released"

        hold_page = torch.as_tensor(token_indices, dtype=torch.int32, device="cpu")
        if hold_page.ndim != 1 or hold_page.numel() != self.page_size:
            raise ValueError("HOLD indices must describe exactly one physical page")
        expected = hold_page[0] + torch.arange(self.page_size, dtype=torch.int32)
        if not torch.equal(hold_page, expected):
            raise ValueError("HOLD indices must form one consecutive physical page")
        self.req_to_token_indexs[self.HOLD_REQUEST_ID].view(-1, self.page_size).copy_(hold_page)

    def alloc(self):
        return self.req_list.alloc()

    def free(self, free_req_indexes: List[int]):
        for req_index in free_req_indexes:
            self.req_list.free(req_index)
        if self.req_list.is_all_free():
            logger.debug(f"freed all request size {self.req_list.can_alloc_size}")

    def free_req(self, free_req_index: int):
        self.free([free_req_index])

    def free_all(self):
        self.req_list = _ReqIndexPool(self.max_request_num)
        return
