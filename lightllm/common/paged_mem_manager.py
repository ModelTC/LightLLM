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
        assert need_size % get_page_size() == 0, "Need size must be a multiple of page size"
        return super().alloc(need_size)

    def free(self, free_index: Union[torch.Tensor, List[int]]):
        page_size = get_page_size()
        if page_size == 1:
            return super().free(free_index)

        if isinstance(free_index, list):
            free_index = torch.tensor(free_index)
        base_free_index = free_index[free_index % page_size == 0]
        if len(base_free_index) == 0:
            return
        token_idxs = base_free_index[:, None] + torch.arange(page_size, device=free_index.device)
        token_idxs = token_idxs.flatten()
        super().free(token_idxs)
        return

    def free_all(self):
        super().free_all()
