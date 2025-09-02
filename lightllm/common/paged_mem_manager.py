import torch
import numpy as np
from .mem_manager import MemoryManager
from typing import List, Union
from lightllm.utils.log_utils import init_logger
from lightllm.utils.envs_utils import get_unique_server_name, get_page_size
from lightllm.server.router.dynamic_prompt.shared_arr import SharedInt
from lightllm.utils.dist_utils import get_current_rank_in_node


def cdiv(a, b):
    return (a + b - 1) // b


logger = init_logger(__name__)


class PagedMemoryManager(MemoryManager):
    def __init__(self, size, dtype, head_num, head_dim, layer_num, always_copy=False, mem_fraction=0.9):
        super().__init__(size, dtype, head_num, head_dim, layer_num, always_copy, mem_fraction)
        page_size = get_page_size()
        self.mem_page_state = torch.arange(
            0, cdiv(self.size, page_size), dtype=torch.int32, device="cpu", requires_grad=False, pin_memory=True
        )
        self.mark_page_start = 0
        self.can_use_page_size = cdiv(self.size, page_size)

    def _init_buffers(self, size, dtype, head_num, head_dim, layer_num):
        self.kv_buffer = torch.empty(
            (layer_num, cdiv(size, get_page_size()) * get_page_size(), 2 * head_num, head_dim),
            dtype=dtype,
            device="cuda",
        )

    # 要求长度必须是page_size的整数倍，page内token索引必须连续
    def check_cache_page_valid(self, values: torch.Tensor):
        end = len(values)
        assert end % self.page_size == 0, "Values length must be a multiple of page size"
        total_pages = end // self.page_size
        for page_idx in range(total_pages):
            values_start = page_idx * self.page_size
            values_end = min((page_idx + 1) * self.page_size, end)
            page_token_idxs = values[values_start:values_end]
            if len(page_token_idxs) > 1:
                expected_idxs = torch.arange(
                    page_token_idxs[0],
                    page_token_idxs[0] + len(page_token_idxs),
                    dtype=page_token_idxs.dtype,
                    device=page_token_idxs.device,
                )
                if not torch.equal(page_token_idxs, expected_idxs):
                    return False
        return True

    def alloc(self, need_size) -> torch.Tensor:
        if self.can_use_page_size < need_size:
            raise RuntimeError(
                f"No available pages for alloc. remaining: {self.can_use_page_size}, needed: {need_size}"
            )
        new_pages = self.mem_page_state[self.mark_page_start : self.mark_page_start + need_size].cuda()
        self.mark_page_start += need_size
        self.can_use_page_size -= need_size
        self.can_use_mem_size -= need_size * get_page_size()
        self.shared_can_use_token_num.set_value(self.can_use_mem_size)
        return new_pages

    def free(self, free_index: Union[torch.Tensor, List[int]]):
        self.can_use_mem_size += len(free_index)
        self.shared_can_use_token_num.set_value(self.can_use_mem_size)

        page_size = get_page_size()
        if isinstance(free_index, list):
            free_index = torch.tensor(free_index, dtype=torch.int32, device="cpu", requires_grad=False, pin_memory=True)

        if len(free_index) == 0:
            return

        base_free_index = free_index[free_index % page_size == 0]
        page_indices = base_free_index // page_size
        for page_idx in sorted(page_indices, reverse=True):  # 逆序放回，保持池的相对顺序
            self.mark_page_start -= 1
            self.mem_page_state[self.mark_page_start] = page_idx
            self.can_use_page_size += 1

        return

    def free_all(self):
        super().free_all()
        page_size = get_page_size()
        self.mark_page_start = 0
        self.can_use_page_size = cdiv(self.size, page_size)
        self.mem_page_state = torch.arange(
            0, cdiv(self.size, page_size), dtype=torch.int32, device="cpu", requires_grad=False, pin_memory=True
        )
