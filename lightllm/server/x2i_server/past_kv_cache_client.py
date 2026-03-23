import ctypes
import torch
import numpy as np
from threading import Lock, Condition
from dataclasses import dataclass
from lightllm.utils.envs_utils import get_env_start_args
from typing import List, Optional, Tuple
from lightllm.utils.log_utils import init_logger
from lightllm.utils.kv_cache_utils import (
    calcu_cpu_cache_meta,
    create_shm_kv_cache_ptr,
    attach_shm_kv_cache_ptr,
    register_shm_ptr_to_pin,
)

logger = init_logger(__name__)


@dataclass
class PastKVCacheItem:
    req_id: int
    token_len: int
    page_indexes: List[int]


class PastKVCacheClient(object):
    """
    This class is responsible for passing kv cache between generation server and model server,
    and manage the shared memory for kv cache.
    """

    def __init__(self, only_create_meta_data: bool, init_shm_data: bool):
        self.args = get_env_start_args()
        # to do here need calcu from from settings.
        self.kv_cache_tensor_meta = calcu_cpu_cache_meta()
        self.page_num: int = self.kv_cache_tensor_meta.page_num
        self.token_page_size: int = self.kv_cache_tensor_meta.token_page_size
        self.allocated_pages_dict: dict[int, PastKVCacheItem] = {}
        self.free_pages: List[int] = list(range(self.page_num))
        self.lock = Lock()
        self.cond = Condition(self.lock)

        if not only_create_meta_data:
            if init_shm_data:
                self._create_shm_cpu_kv_cache()
                self.attach_shm_handle = None
            else:
                self.attach_shm_handle = self._attach_shm_cpu_kv_cache()
                self.attach_shm_handle.wait()
        return

    def allocate_pages(self, req_id: int, need_tokens: int) -> List[int]:
        need_pages = (need_tokens + self.token_page_size - 1) // self.token_page_size
        if need_pages > self.page_num:
            logger.error(
                f"Request {req_id} need {need_tokens} tokens, which requires {need_pages} pages, "
                f"exceeds the total page number {self.page_num}"
            )
            raise ValueError(f"error allocate pages for request {req_id} with {need_tokens} tokens")

        with self.cond:
            while len(self.free_pages) < need_pages:
                self.cond.wait()

            page_indexes, self.free_pages = self.free_pages[:need_pages], self.free_pages[need_pages:]
            self.allocated_pages_dict[req_id] = PastKVCacheItem(
                req_id=req_id, token_len=need_tokens, page_indexes=page_indexes)

            return page_indexes

    def free_pages_by_req_id(self, req_id: int):
        with self.cond:
            item = self.allocated_pages_dict.pop(req_id, None)
            if item is not None:
                self.free_pages.extend(item.page_indexes)
                self.cond.notify_all()

    def get_pages_by_req_id(self, req_id: int) -> Optional[List[int]]:
        with self.lock:
            item = self.allocated_pages_dict.get(req_id, None)
            return item.page_indexes if item is not None else None

    def get_kv_cache_for_x2i(self, page_indexes: List[int], token_num: int) -> Optional[torch.Tensor]:
        if page_indexes is None:
            return None
        assert token_num <= len(page_indexes) * self.token_page_size and \
            token_num > (len(page_indexes) - 1) * self.token_page_size
        (P, L, S, H, D) = self.cpu_kv_cache_tensor[page_indexes].shape
        # (P, L, S, H, D) -> (P, L, S, 2, H // 2, D) -> (2, L, H // 2, P, S, D)
        # -> (2, L, H // 2, P * S, D) -> ( L, 2, H // 2, P * S, D)
        kv = self.cpu_kv_cache_tensor[page_indexes] \
             .view(P, L, S, 2, H // 2, D) \
             .permute(3, 1, 4, 0, 2, 5).contiguous() \
             .view(2, L, H // 2, P * S, D) \
             .permute(1, 0, 2, 3, 4)
        return kv[:, :, :, :token_num, :].contiguous()

    def _create_shm_cpu_kv_cache(self):
        shm_ptr = create_shm_kv_cache_ptr(
            key=self.args.multi_modal_x2i_cache_shm_id, size=self.kv_cache_tensor_meta.calcu_size()
        )
        numpy_array = np.frombuffer(
            memoryview((ctypes.c_uint8 * self.kv_cache_tensor_meta.calcu_size()).from_address(shm_ptr)), dtype=np.uint8
        )
        # 将 NumPy 数组转换为 PyTorch 张量
        shape = (
            self.kv_cache_tensor_meta.page_num,
            self.kv_cache_tensor_meta.layer_num,
            self.kv_cache_tensor_meta.token_page_size,
            self.kv_cache_tensor_meta.num_heads,
            self.kv_cache_tensor_meta.get_merged_head_dim(),
        )
        self.cpu_kv_cache_tensor = (
            torch.from_numpy(numpy_array).view(dtype=self.kv_cache_tensor_meta.data_type).view(shape)
        )
        return

    def _attach_shm_cpu_kv_cache(self):
        shm_ptr = attach_shm_kv_cache_ptr(
            key=self.args.multi_modal_x2i_cache_shm_id, size=self.kv_cache_tensor_meta.calcu_size()
        )
        handle = register_shm_ptr_to_pin(shm_ptr=shm_ptr, size=self.kv_cache_tensor_meta.calcu_size())
        numpy_array = np.frombuffer(
            memoryview((ctypes.c_uint8 * self.kv_cache_tensor_meta.calcu_size()).from_address(shm_ptr)), dtype=np.uint8
        )
        shape = (
            self.kv_cache_tensor_meta.page_num,
            self.kv_cache_tensor_meta.layer_num,
            self.kv_cache_tensor_meta.token_page_size,
            self.kv_cache_tensor_meta.num_heads,
            self.kv_cache_tensor_meta.get_merged_head_dim(),
        )
        self.cpu_kv_cache_tensor = (
            torch.from_numpy(numpy_array).view(dtype=self.kv_cache_tensor_meta.data_type).view(shape)
        )
        assert shm_ptr == self.cpu_kv_cache_tensor.data_ptr()

        return handle
