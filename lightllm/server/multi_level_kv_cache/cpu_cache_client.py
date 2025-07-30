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

    def get_empty_pages(self, num: int) -> List[int]:
        """
        This function is used to get empty pages.
        :param num: the number of pages to get.
        :return: a list contains page ids. page_num <= num.
        """
        assert num >= 0
        if num == 0:
            return []

        ans_list = []
        head = self.page_items.head
        tail = self.page_items.tail
        cur_page: _CpuPageStatus = head.get_next_item()
        while cur_page.self_index != tail.self_index:
            if (cur_page.is_empty() or cur_page.is_ready_recycle()) and cur_page.ref_count == 0:
                ans_list.append(cur_page.self_index)
                next_page = cur_page.get_next_item()
                # del cur_page from list
                cur_page.del_self_from_list()
                cur_page = next_page

                if len(ans_list) >= num:
                    break
            else:
                break

        return ans_list

    def ready_put_back(self, page_index_list: List[int]):
        for page_index in page_index_list:
            cur_page: _CpuPageStatus = self.page_items.get_item_by_index(page_index)
            cur_page.status = cur_page.READY
            self.page_items.add_item_to_tail(index=page_index)
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
    _fields_ = [
        ("status", ctypes.c_int),
        ("ref_count", ctypes.c_int),
    ]

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
