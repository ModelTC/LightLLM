import ctypes
from lightllm.utils.envs_utils import get_env_start_args, get_unique_server_name
from multiprocessing import shared_memory
from typing import List
from lightllm.utils.log_utils import init_logger
from .shm_objs import ShmDict, ShmLinkedList, _LinkedListItem

logger = init_logger(__name__)


class CpuKvCacheClient(object):
    """
    This class is responsible for handling cpu kv cache meta data.
    """

    def __init__(self, init_shm_data: bool):
        self.args = get_env_start_args()
        self.page_num: int = self.args.page_num
        self._create_cpu_status_list(init_shm_data)

    def get_empty_pages(self, page_hashes: List[int]) -> List[int]:
        """
        This function is used to get empty pages.
        :param page_hashes: the hash value of page token.
        :return: a list contains page ids. page_id == -1 is special,
        :means token is found in the cache or the pages is not enough.
        """
        assert len(page_hashes) >= 0
        if len(page_hashes) == 0:
            return []

        exists_mark: List[bool] = []
        for key in page_hashes:
            if self.page_hash_dict.get(key) is not None:
                exists_mark.append(True)
            else:
                exists_mark.append(False)

        page_list = []
        head = self.page_items.head
        tail = self.page_items.tail
        cur_page: _CpuPageStatus = head.get_next_item()

        for hash_key, exist in zip(page_hashes, exists_mark):
            if exist:
                page_list.append(-1)
                continue

            if cur_page.self_index == tail.self_index:
                page_list.append(-1)
            else:
                while cur_page.self_index != tail.self_index:
                    if (cur_page.is_empty() or cur_page.is_ready_recycle()) and cur_page.ref_count == 0:
                        page_list.append(cur_page.self_index)
                        next_page = cur_page.get_next_item()
                        # del cur_page from list
                        cur_page.del_self_from_list()
                        if cur_page.is_ready_recycle():
                            self.page_hash_dict.remove(cur_page.hash_key)
                        cur_page.hash_key = hash_key
                        self.page_hash_dict.put(hash_key, cur_page.self_index)
                        cur_page.status = cur_page.LOADING
                        cur_page = next_page
                        break
                    else:
                        page_list.append(-1)
                        break

        return page_list

    def ready_put_back(self, page_list: List[int]):
        for page_index in page_list:
            cur_page: _CpuPageStatus = self.page_items.get_item_by_index(page_index)
            cur_page.status = cur_page.READY
            self.page_items.add_item_to_tail(index=page_index)
        return

    def query_pages(self, hash_keys: List[int]) -> List[int]:
        """
        query_pages will add _CpuPageStaus ref_count
        """
        page_list = []
        for hash_key in hash_keys:
            page_index = self.page_hash_dict.get(hash_key)
            if page_index is not None:
                page_item: _CpuPageStatus = self.page_items.get_item_by_index(page_index)
                if page_item.is_data_ready():
                    page_list.append(page_index)
                    page_item.ref_count += 1
                    # lru 更新
                    page_item.del_self_from_list()
                    self.page_items.add_item_to_tail(index=page_index)
                else:
                    page_list.append(-1)
                    break
            else:
                page_list.append(-1)
                break
        left_num = len(hash_keys) - len(page_list)
        page_list.extend([-1 for _ in range(left_num)])
        return page_list

    def recycle_pages(self, page_list: List[int]):
        """
        recycle_pages
        """
        for page_index in page_list:
            if page_index != -1:
                page_item: _CpuPageStatus = self.page_items.get_item_by_index(page_index)
                page_item.ref_count -= 1
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
        return


class _CpuPageStatus(_LinkedListItem):
    _pack_ = 4
    _fields_ = [("status", ctypes.c_int), ("ref_count", ctypes.c_int), ("hash_key", ctypes.c_uint64)]

    EMPTY = 0  # 空闲
    LOADING = 1  # 从 gpu buffer 加载到 cpu 的状态，或者是从磁盘加载到 cpu 的状态
    READY = 2  # 数据已经加载到 cpu ok 的状态
    OFFLOADING = 3  # 从 cpu 卸载到 硬盘的状态
    READY_RECYCLE = 4  # 因为卸载到硬盘已经完成，所以可以进行回收使用

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

    def is_offloading(self):
        return self.status == self.OFFLOADING

    def is_ready_recycle(self):
        return self.status == self.READY_RECYCLE

    def is_data_ready(self):
        """
        判断数据是否是填充ok的，可能包含多种状态下属于数据是可填充的状态。
        """
        return self.status >= self.READY
