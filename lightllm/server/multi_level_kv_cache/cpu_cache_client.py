import ctypes
import torch
import numpy as np
from lightllm.utils.envs_utils import get_env_start_args, get_unique_server_name, get_disk_cache_prompt_limit_length
from typing import List, Optional, Tuple
from lightllm.utils.log_utils import init_logger
from .shm_objs import ShmDict, ShmLinkedList, _LinkedListItem, IntList
from lightllm.server.core.objs import AtomicShmLock
from lightllm.utils.kv_cache_utils import (
    calcu_cpu_cache_meta,
    create_shm_kv_cache_ptr,
    attach_shm_kv_cache_ptr,
    register_shm_ptr_to_pin,
)

logger = init_logger(__name__)


class CpuKvCacheClient(object):
    """
    This class is responsible for handling cpu kv cache meta data.
    """

    def __init__(self, only_create_meta_data: bool, init_shm_data: bool):
        self.args = get_env_start_args()
        # to do here need calcu from from settings.
        self.kv_cache_tensor_meta = calcu_cpu_cache_meta()
        self.page_num: int = self.kv_cache_tensor_meta.page_num
        self.lock = AtomicShmLock(lock_name=f"{get_unique_server_name()}_cpu_kv_cache_client_lock")
        self._create_cpu_status_list(init_shm_data)

        if not only_create_meta_data:
            if init_shm_data:
                self._create_shm_cpu_kv_cache()
                self.attach_shm_handle = None
            else:
                self.attach_shm_handle = self._attach_shm_cpu_kv_cache()
        return

    def get_one_empty_page(self, hash_key: int, disk_offload_enable: bool) -> Optional[int]:
        assert self.page_hash_dict.get(hash_key) is None
        head = self.page_items.head
        tail = self.page_items.tail
        cur_page: _CpuPageStatus = head.get_next_item()
        if cur_page.self_index == tail.self_index:
            return None

        if cur_page.can_realloc(disk_offload_enable=disk_offload_enable):
            page_index = cur_page.self_index
            if not cur_page.is_empty():
                self.page_hash_dict.remove(cur_page.hash_key)
            cur_page.hash_key = hash_key
            cur_page.status = cur_page.LOADING
            assert cur_page.ref_count == 0
            cur_page.ref_count += 1
            if cur_page.ref_count == 1:
                cur_page.del_self_from_list()
            self.page_hash_dict.put(hash_key, page_index)
            return page_index
        else:
            return None

    def allocate_one_page(
        self, page_items: List[_LinkedListItem], hash_key: int, disk_offload_enable: bool
    ) -> Tuple[Optional[int], bool]:
        page_index = self.page_hash_dict.get(hash_key)
        if page_index is not None:
            page_item: _CpuPageStatus = page_items[page_index]
            page_item.ref_count += 1
            if page_item.ref_count == 1:
                page_item.del_self_from_list()
            if page_item.is_data_ready():
                return page_index, True
            else:
                return page_index, False
        else:
            page_index = self.get_one_empty_page(hash_key=hash_key, disk_offload_enable=disk_offload_enable)
            if page_index is not None:
                return page_index, False
            else:
                return None, False

    def allocate_pages(self, hash_keys: List[int], disk_offload_enable: bool) -> Tuple[List[int], List[bool]]:
        """
        allocate_pages will add _CpuPageStaus ref_count
        """
        page_list = []
        ready_list = []
        page_items = self.page_items.linked_items
        for hash_key in hash_keys:
            page_index, ready = self.allocate_one_page(
                page_items=page_items, hash_key=hash_key, disk_offload_enable=disk_offload_enable
            )
            if page_index is not None:
                page_list.append(page_index)
                ready_list.append(ready)
            else:
                page_list.append(-1)
                ready_list.append(False)
                break

        left_num = len(hash_keys) - len(page_list)
        page_list.extend([-1 for _ in range(left_num)])
        ready_list.extend([False for _ in range(left_num)])
        return page_list, ready_list

    def update_pages_status_to_ready(
        self,
        page_list: List[int],
        deref: bool = True,
        disk_offload_enable: bool = False,
    ):
        offload_candidates: List[int] = []
        page_items = self.page_items.linked_items
        not_exist_none_page = True
        for page_index in page_list:
            if page_index != -1:
                cur_page: _CpuPageStatus = page_items[page_index]
                if cur_page.status < _CpuPageStatus.READY:
                    cur_page.status = _CpuPageStatus.READY

                if deref:
                    assert cur_page.ref_count > 0
                    cur_page.ref_count -= 1
                    if cur_page.ref_count == 0:
                        # 放回 LRU 列表尾部
                        self.page_items.add_item_to_tail(cur_page.self_index)

                # 全部落盘，已落盘前缀部分会在落盘中自动剔除
                if disk_offload_enable and not_exist_none_page:
                    offload_candidates.append(cur_page.self_index)

            else:
                not_exist_none_page = False

        # 控制prompt长度，较短的prompt不进行disk offload
        limit_length = get_disk_cache_prompt_limit_length()

        if (
            disk_offload_enable
            and offload_candidates
            and len(page_list) * self.args.cpu_cache_token_page_size >= limit_length
        ):
            # 加引用计数，落盘成功后再减掉
            for offload_page_index in offload_candidates:
                offload_page_item: _CpuPageStatus = page_items[offload_page_index]
                offload_page_item.ref_count += 1
                if offload_page_item.ref_count == 1:
                    # 从 LRU 列表中移除
                    offload_page_item.del_self_from_list()
            # 写入到 offload_page_indexes 中的数据是分组的，其中
            # 开头的元素标记后续多少个元素是一组，便于读取时进行分组处理
            # 写入数据为 group_page_size, page_index1, page_index2, ...
            self.offload_page_indexes.add_item(len(offload_candidates))
            self.offload_page_indexes.add_items(offload_candidates)
        return

    def query_one_page(self, hash_key: int) -> Tuple[Optional[int], bool]:
        """
        返回的cpu page必然是数据ready可以被复用的。
        """
        page_index = self.page_hash_dict.get(hash_key)
        if page_index is not None:
            page_item: _CpuPageStatus = self.page_items.get_item_by_index(page_index)
            if page_item.is_data_ready():
                page_item.ref_count += 1
                if page_item.ref_count == 1:
                    page_item.del_self_from_list()
                return page_index, True
            else:
                if page_item.ref_count == 0:
                    # lru 更新
                    page_item.del_self_from_list()
                    self.page_items.add_item_to_tail(index=page_index)
                return None, False
        else:
            return None, False

    def check_allpages_ready(self, page_list: List[int]) -> bool:
        page_items = self.page_items.linked_items
        for page_index in page_list:
            if page_index == -1:
                continue
            page_item: _CpuPageStatus = page_items[page_index]
            if not page_item.is_data_ready():
                return False
        return True

    def deref_pages(self, page_list: List[int]):
        """
        deref_pages
        """
        page_items = self.page_items.linked_items
        for page_index in page_list:
            if page_index != -1:
                page_item: _CpuPageStatus = page_items[page_index]
                assert page_item.ref_count > 0
                page_item.ref_count -= 1
                if page_item.ref_count == 0:
                    # 放回 LRU 列表头部
                    self.page_items.add_item_to_tail(page_item.self_index)
        return

    def deref_one_page(self, page_index: int):
        page_item: _CpuPageStatus = self.page_items.get_item_by_index(page_index)
        assert page_item.ref_count > 0
        page_item.ref_count -= 1
        if page_item.ref_count == 0:
            # 放回 LRU 列表头部
            self.page_items.add_item_to_tail(page_item.self_index)
        return

    def get_pages_to_offloading(self) -> List[List[int]]:
        page_list = self.offload_page_indexes.pop_all_item()
        groups: List[List[int]] = []

        if page_list is None:
            return groups

        page_items = self.page_items.linked_items
        index = 0
        while index < len(page_list):
            group_size = page_list[index]
            groups.append(page_list[index + 1 : index + 1 + group_size])
            for page_index in groups[-1]:
                page_item: _CpuPageStatus = page_items[page_index]
                assert page_item.is_ready()

            index = index + 1 + group_size

        return groups

    def recycle_pages(self, page_list: List[int]):
        """
        当从硬盘cache中读取数据失败时,调用此函数回收页面
        """
        page_items = self.page_items.linked_items
        for page_index in page_list:
            if page_index == -1:
                continue
            cur_page: _CpuPageStatus = page_items[page_index]

            if cur_page.ref_count > 0:
                cur_page.ref_count -= 1
                if cur_page.ref_count == 0:
                    if cur_page.is_loading():
                        existing_index = self.page_hash_dict.get(cur_page.hash_key)
                        assert existing_index is not None and existing_index == cur_page.self_index
                        self.page_hash_dict.remove(cur_page.hash_key)
                        cur_page.hash_key = 0
                        cur_page.status = _CpuPageStatus.EMPTY
                        self.page_items.add_item_to_head(cur_page.self_index)
                    else:
                        self.page_items.add_item_to_tail(cur_page.self_index)

        return

    def _create_cpu_status_list(self, init_shm_data: bool):
        self.page_items = ShmLinkedList(
            name=f"{get_unique_server_name()}_cpu_kv_cache_page_items",
            item_class=_CpuPageStatus,
            capacity=self.page_num,
            init_shm_data=init_shm_data,
        )
        self.page_hash_dict = ShmDict(
            name=f"{get_unique_server_name()}_cpu_kv_cache_hash",
            capacity=self.page_num * 2,
            init_shm_data=init_shm_data,
        )
        self.offload_page_indexes = IntList(
            name=f"{get_unique_server_name()}_cpu_kv_cache_offload_page_indexes",
            capacity=self.page_num * 2,
            init_shm_data=init_shm_data,
        )
        return

    def _create_shm_cpu_kv_cache(self):
        shm_ptr = create_shm_kv_cache_ptr()
        numpy_array = np.frombuffer(
            memoryview((ctypes.c_uint8 * self.kv_cache_tensor_meta.calcu_size()).from_address(shm_ptr)), dtype=np.uint8
        )
        # 将 NumPy 数组转换为 PyTorch 张量
        shape = (
            self.kv_cache_tensor_meta.page_num,
            self.kv_cache_tensor_meta.layer_num,
            self.kv_cache_tensor_meta.token_page_size,
            self.kv_cache_tensor_meta.num_heads,
            self.kv_cache_tensor_meta.head_dim,
        )
        self.cpu_kv_cache_tensor = torch.from_numpy(numpy_array).view(dtype=torch.bfloat16).view(shape)
        return

    def _attach_shm_cpu_kv_cache(self):
        shm_ptr = attach_shm_kv_cache_ptr()
        handle = register_shm_ptr_to_pin(shm_ptr=shm_ptr, size=self.kv_cache_tensor_meta.calcu_size())
        numpy_array = np.frombuffer(
            memoryview((ctypes.c_uint8 * self.kv_cache_tensor_meta.calcu_size()).from_address(shm_ptr)), dtype=np.uint8
        )
        shape = (
            self.kv_cache_tensor_meta.page_num,
            self.kv_cache_tensor_meta.layer_num,
            self.kv_cache_tensor_meta.token_page_size,
            self.kv_cache_tensor_meta.num_heads,
            self.kv_cache_tensor_meta.head_dim,
        )
        self.cpu_kv_cache_tensor = torch.from_numpy(numpy_array).view(dtype=torch.bfloat16).view(shape)
        assert shm_ptr == self.cpu_kv_cache_tensor.data_ptr()

        # test code
        # self.cpu_kv_cache_tensor = torch.zeros_like(self.cpu_kv_cache_tensor, device="cpu", pin_memory=True)
        # self.cpu_kv_cache_tensor = torch.zeros_like(self.cpu_kv_cache_tensor, device="cuda")
        return handle


class _CpuPageStatus(_LinkedListItem):
    _pack_ = 4
    _fields_ = [("status", ctypes.c_int), ("ref_count", ctypes.c_int), ("hash_key", ctypes.c_uint64)]

    EMPTY = 0  # 空闲
    LOADING = 1  # 从 gpu buffer 加载到 cpu 的状态，或者是从磁盘加载到 cpu 的状态
    READY = 2  # 数据已经加载到 cpu ok 的状态

    def __init__(self):
        self.init()

    def init(self):
        super().init()
        self.ref_count = 0
        self.status = self.EMPTY
        self.hash_key = 0
        return

    def is_empty(self):
        return self.status == self.EMPTY

    def is_loading(self):
        return self.status == self.LOADING

    def is_ready(self):
        return self.status == self.READY

    def is_data_ready(self):
        """
        判断数据是否是填充ok的，可能包含多种状态下属于数据是可填充的状态。
        """
        return self.status >= self.READY

    def can_realloc(self, disk_offload_enable: bool):
        return (self.is_empty() or self.is_data_ready()) and self.ref_count == 0
