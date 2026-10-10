from typing import List

import numpy as np

import torch
from triton.runtime import driver

from lightllm.common.kv_cache_mem_manager import MemoryManager
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
    def __init__(self, max_request_num, max_sequence_length, mem_manager: MemoryManager):
        from .req_sampling_params import ReqSamplingParamsManager

        # 这里对最大请求数量的管理在默认上多申请了一个，主要是 index 为 max_request_num 代表
        # 的这个请求管理 id， 主要是为了兼容 DP 运行模式下，让各个 DP 能 padding 到 DP 中最大
        # 的那个batch size 进行运行，所有 padding 的请求都会使用预留的这个请求管理 id 进行处理
        # 这样让 DP 的实现更为简化一些。
        self.req_list = _ReqIndexPool(max_request_num)
        page_size = get_env_start_args().page_size
        max_sequence_length = (max_sequence_length + page_size - 1) // page_size * page_size
        self.req_to_token_indexs = torch.zeros(
            (max_request_num + 1, max_sequence_length), dtype=torch.int32, device="cuda"
        )
        self._cpu_pages = np.empty((max_request_num + 1, max_sequence_length // page_size), dtype=np.int32)
        self.token_indexes_ready = torch.cuda.Event()
        self._token_indexes_stream = None
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
        self._cpu_pages[self.HOLD_REQUEST_ID].fill(self.mem_manager.HOLD_TOKEN_MEMINDEXES[0])
        self.req_to_token_indexs[self.HOLD_REQUEST_ID].copy_(
            self.get_cpu_token_indexes(self.HOLD_REQUEST_ID, 0, self.req_to_token_indexs.shape[1])
        )

    def get_cpu_page_bases(self, req_idx, start, end):
        """Return a CPU view of page bases for the token range [start, end)."""
        page_size = self.mem_manager.page_size
        assert start % page_size == end % page_size == 0
        return self._cpu_pages[req_idx, start // page_size : end // page_size]

    def get_cpu_token_indexes(self, req_idx, start, end):
        """Materialize owned token indexes only for cache insertion and PD metadata."""
        page_size = self.mem_manager.page_size
        first, offset = divmod(start, page_size)
        bases = self._cpu_pages[req_idx, first : (end + page_size - 1) // page_size]
        indexes = self.mem_manager.allocator.expand_pages(bases)
        return torch.from_numpy(indexes[offset : offset + end - start])

    def write_token_index_prefix(self, req_idx, start, indexes, req_prefix_lens):
        """Update CPU prefix mappings and collect pending GPU copies."""
        page_size = self.mem_manager.page_size
        assert start == 0
        size = indexes.numel()
        assert size % page_size == 0
        self._cpu_pages[req_idx, start // page_size : (start + size) // page_size] = indexes.numpy()[::page_size]
        if size:
            req_prefix_lens[req_idx] = max(size, req_prefix_lens.get(req_idx, 0))

    def copy_token_index_prefixes_to_gpu(self, req_prefix_lens):
        """Copy the collected CPU token-index prefixes on the current CUDA stream."""
        if req_prefix_lens:
            updates, offset = [], 0
            for row, size in req_prefix_lens.items():
                updates.append((row, 0, size, offset))
                offset += size // self.mem_manager.page_size
            self._copy_token_indexes_to_gpu(updates)

    def alloc_req_pages(self, allocations, radix_cache=None):
        """Reserve pages once; CPU and GPU tables consume the same page-base assignment."""
        need = sum(size for _, size in allocations)
        if radix_cache is not None:
            radix_cache.free_radix_cache_to_get_enough_token(need)
        page_size = self.mem_manager.page_size
        pages = self.mem_manager.allocator._alloc_pages(need)
        updates, offset = [], 0
        for req, size in allocations:
            start = req.hold_kv_len
            if size == page_size:
                self._cpu_pages[req.req_idx, start // page_size] = pages[offset]
            else:
                self._cpu_pages[req.req_idx, start // page_size : (start + size) // page_size] = pages[
                    offset : offset + size // page_size
                ]
            updates.append((req.req_idx, start, size, offset))
            req.hold_kv_len += size
            offset += size // page_size
        self._copy_token_indexes_to_gpu(updates, pages)

    def alloc_token_indexes(self, req, end, radix_cache=None):
        """Return write slots for [cur_kv_len, end), reusing any existing page tail."""
        size = self.get_need_alloc_token_num(req, end)
        if size:
            self.alloc_req_pages([(req, size)], radix_cache)
        return self.get_cpu_token_indexes(req.req_idx, req.cur_kv_len, end)

    def _copy_token_indexes_to_gpu(self, updates, pages=None):
        """Snapshot compact pages, using direct DMA for small batches and GPU expansion otherwise."""
        from lightllm.common.basemodel.triton_kernel.copy_kv_index_to_req import update_req_token_indexes

        page_size = self.mem_manager.page_size
        if pages is None:
            parts = [
                self._cpu_pages[row, start // page_size : (start + size) // page_size]
                for row, start, size, _ in updates
            ]
            pages = parts[0] if len(parts) == 1 else np.concatenate(parts)
        if not len(pages):
            return
        packed = len(updates) > 4
        header = 4 * len(updates) if packed else 0
        packet = torch.empty(
            header + len(pages) * (1 if packed else page_size), dtype=torch.int32, device="cpu", pin_memory=True
        )
        data = packet.numpy()
        if packed:
            meta = data[:header].reshape(-1, 4)
            meta[:] = updates
            meta[:, 3] += header
            data[header:] = pages
        else:
            self.mem_manager.allocator.expand_pages(pages, data)
        stream = self.wait_token_indexes()
        if packed:
            update_req_token_indexes(
                self.req_to_token_indexs,
                packet.cuda(non_blocking=True),
                len(updates),
                max(x[2] for x in updates),
                page_size,
            )
        else:
            table = self.req_to_token_indexs
            row_stride, token_stride = table.stride()
            base = table.storage_offset()
            for row, start, size, offset in updates:
                table.as_strided((size,), (token_stride,), base + row * row_stride + start * token_stride).copy_(
                    packet if len(updates) == 1 else packet[offset * page_size : offset * page_size + size],
                    non_blocking=True,
                )
        self.token_indexes_ready.record(stream)
        self._token_indexes_stream = stream

    def get_need_alloc_token_num(self, req, target):
        """Return additional page-aligned KV capacity, measured in token slots."""
        page_size = self.mem_manager.page_size
        size = max(0, (target + page_size - 1) // page_size * page_size - req.hold_kv_len)
        assert size % page_size == 0
        return size

    def classify_reqs_and_alloc_kv(self, candidates, available, batch_max_tokens, radix_cache=None):
        """按优先级顺序接纳请求，再一次性预留所有必需的页。

        prefill 和 decode 两组请求都会跨调度轮次保留已分配的页。
        decode 的可选额外预留仅使用满足所有必要分配后剩余的空闲容量。
        """
        prefill, decode, rejected, allocations, headroom = [], [], [], [], []
        prefill_tokens = required = 0
        page_size = self.mem_manager.page_size
        for req, is_decode, tokens in candidates:
            if not is_decode and prefill_tokens + tokens > batch_max_tokens:
                continue

            target = req.cur_kv_len + tokens
            size = self.get_need_alloc_token_num(req, target) if target > req.hold_kv_len else 0
            if size > available:
                rejected.append((req, is_decode))
                continue

            (decode if is_decode else prefill).append(req)
            available -= size
            if not is_decode:
                prefill_tokens += tokens
            if size:
                allocation = [req, size]
                allocations.append(allocation)
                required += size
                if is_decode and page_size < 8:
                    headroom.append(allocation)

        extra_budget = max(0, self.mem_manager.allocator.can_use_mem_size - required)
        for allocation in headroom:
            extra = min(8, extra_budget) // page_size * page_size
            allocation[1] += extra
            extra_budget -= extra
        if allocations:
            self.alloc_req_pages(allocations, radix_cache)
        return prefill, decode, rejected, prefill_tokens

    def wait_token_indexes(self):
        # Reuse the writer's Python Stream; query only the raw handle on the common path.
        stream = self._token_indexes_stream
        if stream is None or driver.active.get_current_stream(stream.device_index) != stream.cuda_stream:
            stream = torch.cuda.current_stream()
            self.token_indexes_ready.wait(stream)
        return stream

    def alloc(self):
        return self.req_list.alloc()

    def free(self, free_req_indexes: List[int], free_page_index):
        for req_index in free_req_indexes:
            self.req_list.free(req_index)

        if self.req_list.is_all_free():
            logger.debug(f"freed all request size {self.req_list.can_alloc_size}")
        self.mem_manager.allocator.release_pages(free_page_index)

    def free_req(self, free_req_index: int):
        self.req_list.free(free_req_index)
        if self.req_list.is_all_free():
            logger.debug(f"freed all request size {self.req_list.can_alloc_size}")
        return

    def free_pages(self, free_page_index):
        self.mem_manager.allocator.release_pages(free_page_index)
        return

    def free_all(self):
        self.req_list = _ReqIndexPool(self.max_request_num)
        return
