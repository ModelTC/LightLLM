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


class PageSizeVariableMemoryManager(MemoryManager):
    def __init__(self, size, dtype, head_num, head_dim, layer_num, always_copy=False, mem_fraction=0.9):
        super().__init__(size, dtype, head_num, head_dim, layer_num, always_copy, mem_fraction)
        self.req_to_page_indexs = None
        page_size = get_page_size()
        self.page_idx_pool = torch.arange(
            0, cdiv(self.size, page_size), dtype=torch.int32, device="cpu", requires_grad=False, pin_memory=True
        )
        self.mark_page_start = 0
        self.can_use_page_size = cdiv(self.size, page_size)

        rank_in_node = get_current_rank_in_node()
        self.shared_can_use_page_num = SharedInt(
            f"{get_unique_server_name()}_mem_manger_can_use_page_num_{rank_in_node}"
        )
        self.shared_can_use_page_num.set_value(self.can_use_page_size)

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

    def set_prefix_cache_to_req(self, req_idx: int, start: int, end: int, values: torch.Tensor):
        # assert self.check_cache_page_valid(values), "Values must be valid for page size"
        page_size = get_page_size()
        self.req_to_page_indexs[req_idx, start // page_size : end // page_size] = values[::page_size] // page_size
        self.req_to_token_indexs[req_idx, start:end] = values

    def expand_by_page_size(self, b_token_len, page_size):
        # 将seq_len按page整数倍展开，例如seq_len = [9,9,9] -> page_len = [4,4,1,4,4,1,4,4,1], page_size = 4
        b_page_len = cdiv(b_token_len, page_size)
        need_pages_num = b_page_len.sum()
        p_token_len = torch.full((need_pages_num,), page_size, dtype=b_token_len.dtype, device=b_token_len.device)
        cumsum_pages = torch.cumsum(b_page_len, dim=0)
        last_page_positions = cumsum_pages - 1
        remainders = b_token_len - (b_page_len - 1) * page_size
        p_token_len[last_page_positions] = remainders
        return need_pages_num, b_page_len, p_token_len

    def get_paged_token_indexs(self, b_req_idx, page_size, b_seq_len, b_ready_cache_len, is_prefill):
        if is_prefill:
            b_req_idx = b_req_idx.cuda()
            b_seq_len = b_seq_len.cuda()
            b_ready_cache_len = b_ready_cache_len.cuda()

            b_token_len = b_seq_len - b_ready_cache_len
            total_pages_needed, b_page_len, p_token_len = self.expand_by_page_size(b_token_len, page_size)
            if self.can_use_page_size < total_pages_needed:
                raise RuntimeError(
                    f"No available pages for alloc. remaining: {self.can_use_page_size}, needed: {total_pages_needed}"
                )

            allocated_pages = self.page_idx_pool[
                self.mark_page_start : self.mark_page_start + total_pages_needed
            ].cuda()

            def get_offsets_by_length(b_len, max_len):
                # 例：b_len = [3,4,5] -> [0,1,2,0,1,2,3,0,1,2,3,4]
                offsets = torch.arange(max_len, dtype=b_len.dtype, device=b_len.device)
                offset_mask = offsets.unsqueeze(0) < b_len.unsqueeze(1)
                return torch.masked_select(offsets, offset_mask)

            page_offsets = get_offsets_by_length(b_page_len, b_page_len.max())
            token_offsets = get_offsets_by_length(p_token_len, page_size)

            # 更新req_to_page_indexs, b_ready_cache_len必整除page_size
            page_starts = b_ready_cache_len // page_size
            req_id = torch.repeat_interleave(
                torch.arange(len(b_req_idx), dtype=b_token_len.dtype, device=b_token_len.device), b_page_len
            )
            self.req_to_page_indexs[b_req_idx[req_id], page_starts[req_id] + page_offsets] = allocated_pages

            self.mark_page_start += total_pages_needed
            self.can_use_page_size -= total_pages_needed
            page_bases = allocated_pages * page_size
            return torch.repeat_interleave(page_bases, p_token_len) + token_offsets
        else:
            b_seq_len = b_seq_len.cuda()
            b_req_idx = b_req_idx.cuda()
            need_new_page_mask = (b_seq_len - 1) % page_size == 0
            new_pages_num = need_new_page_mask.sum()
            if self.can_use_page_size < new_pages_num:
                raise RuntimeError(
                    f"No available pages for alloc. remaining: {self.can_use_page_size}, needed: {new_pages_num}"
                )

            token_idxs = torch.zeros_like(b_seq_len, device=b_seq_len.device)
            if new_pages_num > 0:
                new_pages = self.page_idx_pool[self.mark_page_start : self.mark_page_start + new_pages_num].cuda()
                self.mark_page_start += new_pages_num
                self.can_use_page_size -= new_pages_num
                token_idxs[need_new_page_mask] = new_pages * page_size

                # 需要更新req_to_page_indexs
                new_page_req_indices = b_req_idx[need_new_page_mask]
                page_positions = (b_seq_len[need_new_page_mask] - 1) // page_size
                self.req_to_page_indexs[new_page_req_indices, page_positions] = new_pages

            mask = ~need_new_page_mask
            if mask.any():
                seq_lens = b_seq_len[mask]
                token_idxs[mask] = (
                    self.req_to_token_indexs[b_req_idx[mask], seq_lens - 2] // page_size * page_size
                    + (seq_lens - 1) % page_size
                )
        return token_idxs

    def alloc(self, need_size, b_req_idx, b_seq_len, b_ready_cache_len=None, is_prefill=False) -> torch.Tensor:
        page_size = get_page_size()
        token_idxs = self.get_paged_token_indexs(b_req_idx, page_size, b_seq_len, b_ready_cache_len, is_prefill)
        self.can_use_mem_size -= need_size
        self.shared_can_use_token_num.set_value(self.can_use_mem_size)
        self.shared_can_use_page_num.set_value(self.can_use_page_size)
        return token_idxs

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
            self.page_idx_pool[self.mark_page_start] = page_idx
            self.can_use_page_size += 1
        self.shared_can_use_page_num.set_value(self.can_use_page_size)

        return

    def free_all(self):
        super().free_all()
        page_size = get_page_size()
        self.mark_page_start = 0
        self.can_use_page_size = cdiv(self.size, page_size)
        self.shared_can_use_page_num.set_value(self.can_use_page_size)
        self.page_idx_pool = torch.arange(
            0, cdiv(self.size, page_size), dtype=torch.int32, device="cpu", requires_grad=False, pin_memory=True
        )
