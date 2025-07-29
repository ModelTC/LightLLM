import ctypes
from lightllm.utils.envs_utils import get_env_start_args, get_unique_server_name
from multiprocessing import shared_memory
from typing import List
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class CpuKvCacheClient(object):
    """
    This class is responsible for handling cpu kv cache meta data.
    """

    def __init__(self):
        self.args = get_env_start_args()
        # add head and tail node
        self.page_num: int = self.args.page_num + 2
        self._create_cpu_status_list()

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
        cur_page = self.pages[self.head.next_index]
        while cur_page.self_index != self.page_num - 1:
            if (cur_page.is_empty() or cur_page.is_ready_recycle()) and cur_page.ref_count == 0:
                ans_list.append(cur_page.self_index)
                # del cur_page from list
                pre_node = self.pages[cur_page.pre_index]
                next_node = self.pages[cur_page.next_index]
                pre_node.next_index = next_node.self_index
                next_node.pre_index = pre_node.self_index

                cur_page = self.pages[cur_page.next_index]
                if len(ans_list) >= num:
                    break
            else:
                break

        return [e - 1 for e in ans_list]

    def ready_put_back(self, page_index_list: List[int]):
        for index in page_index_list:
            page_index = index + 1
            cur_page = self.pages[page_index]
            cur_page.status = cur_page.READY
            # 将节点加入到链表中
            pre_node = self.pages[self.tail.pre_index]
            pre_node.next_index = page_index
            cur_page.pre_index = pre_node.self_index
            cur_page.next_index = self.tail.self_index
            self.tail.pre_index = page_index
        return

    def _create_cpu_status_list(self):
        class_size = ctypes.sizeof(_CpuPageStatus)
        byte_size = class_size * self.page_num

        shm_name = f"{get_unique_server_name()}_cpu_kv_cache_meta"
        shm = _create_shm(name=shm_name, byte_size=byte_size)
        self.shm = shm

        if self.shm.size != byte_size:
            logger.info(f"size not same, unlink lock shm {self.shm.name} and create again")
            self.shm.close()
            self.shm.unlink()
            self.shm = None
            self.shm = _create_shm(name=shm_name, byte_size=byte_size)

        # 构建链表关系。
        self.pages: List[_CpuPageStatus] = (_CpuPageStatus * self.page_num).from_buffer(self.shm.buf)
        for e in self.pages:
            e.init()

        self.head = self.pages[0]
        self.head.self_index = 0
        self.head.next_index = 1
        self.tail = self.pages[self.page_num - 1]
        self.tail.self_index = self.page_num - 1
        self.tail.pre_index = self.page_num - 2

        for i in range(1, self.page_num - 1):
            status_item = self.pages[i]
            status_item.self_index = i
            status_item.pre_index = i - 1
            status_item.next_index = i + 1
        return


class _CpuPageStatus(ctypes.Structure):
    _pack_ = 4
    _fields_ = [
        ("self_index", ctypes.c_int),
        ("pre_index", ctypes.c_int),
        ("next_index", ctypes.c_int),
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
        self.self_index = -1
        self.pre_index = -1
        self.next_index = -1
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


class ShmDict(object):
    def __init__(self):
        self.args = get_env_start_args()
        self.capacity: int = self.args.page_num * 2

    def _create_hash_table_list(self):
        class_size = ctypes.sizeof(_HashItem)
        byte_size = class_size * self.capacity

        shm_name = f"{get_unique_server_name()}_cpu_kv_cache_hash_table"
        shm = _create_shm(name=shm_name, byte_size=byte_size)
        self.shm = shm

        if self.shm.size != byte_size:
            logger.info(f"size not same, unlink lock shm {self.shm.name} and create again")
            self.shm.close()
            self.shm.unlink()
            self.shm = None
            self.shm = _create_shm(name=shm_name, byte_size=byte_size)
        # 构建 hash table 表
        self.item_list: List[_HashItem] = (_HashItem * self.capacity).from_buffer(self.shm.buf)
        for e in self.item_list:
            e.init()
        return


class _HashItem(ctypes.Structure):
    _pack_ = 4
    _fields_ = [
        ("exist", ctypes.c_uint32),  # 0 not exist, 1 exist
        ("key", ctypes.c_uint64),
        ("value", ctypes.c_int),
    ]

    def __init__(self):
        self.init()

    def init(self):
        self.exist = 0
        self.key = -1
        self.value = -1


def _create_shm(name: str, byte_size: int):
    try:
        shm = shared_memory.SharedMemory(name=name, create=True, size=byte_size)
        logger.info(f"create lock shm {name}")
    except:
        shm = shared_memory.SharedMemory(name=name, create=False, size=byte_size)
        logger.info(f"link lock shm {name}")
    return shm
