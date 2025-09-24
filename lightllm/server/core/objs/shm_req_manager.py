import ctypes
import numpy as np
from lightllm.utils.envs_utils import get_unique_server_name
from multiprocessing import shared_memory
from lightllm.utils.log_utils import init_logger
from .req import Req, ChunkedPrefillReq, TokenHealingReq
from .shm_array import ShmArray
from .atomic_array_lock import AtomicShmArrayLock, AtomicLockItem
from .atomic_lock import AtomicShmLock
from .start_args_type import StartArgs
from typing import List
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.shm_utils import create_or_link_shm

logger = init_logger(__name__)


class ShmReqManager:
    def __init__(self):
        self.req_class: Req.__class__ = self.get_req_class_type()
        class_size = ctypes.sizeof(self.req_class)
        self.max_req_num = self.get_max_req_num()
        self.req_shm_byte_size = class_size * self.max_req_num

        self.init_reqs_shm()
        self.init_to_req_objs()
        self.init_to_req_locks()
        self.init_manager_lock()
        self.init_alloc_state_shm()
        return

    def get_req_class_type(self):
        args: StartArgs = get_env_start_args()
        if args.token_healing_mode:
            return TokenHealingReq

        return ChunkedPrefillReq

    def get_max_req_num(self):
        args: StartArgs = get_env_start_args()
        return args.running_max_req_size

    def init_reqs_shm(self):
        self._init_reqs_shm()

        if self.reqs_shm.size != self.req_shm_byte_size:
            logger.info(f"size not same, unlink lock shm {self.reqs_shm.name} and create again")
            self.reqs_shm.close()
            self.reqs_shm.unlink()
            self.reqs_shm = None
            self._init_reqs_shm()

    def _init_reqs_shm(self):
        shm_name = f"{get_unique_server_name()}_req_shm_total"
        self.reqs_shm = create_or_link_shm(shm_name, self.req_shm_byte_size)
        return

    def init_to_req_objs(self):
        self.reqs: List[Req] = (self.req_class * self.max_req_num).from_buffer(self.reqs_shm.buf)
        for i in range(self.max_req_num):
            self.reqs[i].ref_count = 0
            self.reqs[i].index_in_shm_mem = i
        return

    def init_to_req_locks(self):
        array_lock_name = f"{get_unique_server_name()}_array_reqs_lock"
        self.reqs_lock = AtomicShmArrayLock(array_lock_name, self.max_req_num)
        return

    def get_req_lock_by_index(self, req_index_in_mem: int) -> AtomicLockItem:
        return self.reqs_lock.get_lock_context(req_index_in_mem)

    def init_manager_lock(self):
        lock_name = f"{get_unique_server_name()}_shm_reqs_manager_lock"
        self.manager_lock = AtomicShmLock(lock_name)
        return

    def init_alloc_state_shm(self):
        shm_name = f"{get_unique_server_name()}_req_alloc_states"
        req_link_list_name = f"{get_unique_server_name()}_req_linked_states"
        self.linked_req_manager = ReqLinkedListManager(req_link_list_name, self.max_req_num)
        self.alloc_state_shm = ShmArray(shm_name, (self.max_req_num,), np.int32)
        self.alloc_state_shm.create_shm()
        self.alloc_state_shm.arr[:] = 0
        # 用来做为每个进程独立的状态管理，用于申请和
        self.proc_private_get_state = np.zeros(shape=(self.max_req_num,), dtype=np.int32)
        return

    # alloc_req_index 和 release_req_index 是分配资源时使用的接口。
    # 只有管理请求申请和释放的首节点才能调用这个接口。
    def alloc_req_index(self):
        with self.manager_lock:
            idx = self.linked_req_manager.alloc()
            if idx is not None:
                assert self.alloc_state_shm.arr[idx] == 0
                self.alloc_state_shm.arr[idx] = 1
            return idx
        return None

    async def async_alloc_req_index(self):
        return self.alloc_req_index()

    def release_req_index(self, req_index_in_mem):
        assert req_index_in_mem < self.max_req_num
        with self.manager_lock:
            assert self.alloc_state_shm.arr[req_index_in_mem] == 1
            self.alloc_state_shm.arr[req_index_in_mem] = 0
            self.linked_req_manager.free(req_index_in_mem)

        if np.sum(self.alloc_state_shm.arr) == 0 and self.linked_req_manager.test_is_full():
            logger.info("all shm req has been release ok")
        return

    async def async_release_req_index(self, req_index_in_mem):
        return self.release_req_index(req_index_in_mem)

    # get_req_obj_by_index 和 put_back_req_obj 是 分配好后，进行对象获取和
    # 管理的接口，主要是要进行引用计数的管理。
    def get_req_obj_by_index(self, req_index_in_mem):
        assert req_index_in_mem < self.max_req_num
        assert self.proc_private_get_state[req_index_in_mem] == 0
        ans: Req = self.reqs[req_index_in_mem]
        with self.get_req_lock_by_index(req_index_in_mem):
            ans.ref_count = ans.ref_count + 1
        self.proc_private_get_state[req_index_in_mem] = 1
        return ans

    async def async_get_req_obj_by_index(self, req_index_in_mem):
        return self.get_req_obj_by_index(req_index_in_mem)

    def put_back_req_obj(self, req: Req):
        req_index_in_mem = req.index_in_shm_mem
        assert req_index_in_mem < self.max_req_num
        assert self.proc_private_get_state[req_index_in_mem] == 1
        with self.get_req_lock_by_index(req_index_in_mem):
            req.ref_count = req.ref_count - 1
        self.proc_private_get_state[req_index_in_mem] = 0

    async def async_put_back_req_obj(self, req: Req):
        return self.put_back_req_obj(req)

    def __del__(self):
        self.reqs = None


class ReqLinkedListManager:
    NEXT_INDEX = 0  # 仅保留指向下一个可用索引的指针

    def __init__(self, name: str, size: int) -> None:
        self.size = size + 1  # +1 , 0 号节点用于链表头
        self._shm_array = ShmArray(name, shape=(self.size, 1), dtype=np.int64)  # 只需一列
        self._shm_array.create_shm()
        self._values = self._shm_array.arr

        self._initialize_values()

    def _initialize_values(self) -> None:
        # Initialize next pointers
        self._values[0 : self.size - 1, self.NEXT_INDEX] = np.arange(1, self.size)
        self._values[self.size - 1, self.NEXT_INDEX] = -1  # 最后一个指向 -1（表示空）

    def alloc(self) -> int:
        """Allocate an index from the linked list."""
        if self._values[0, self.NEXT_INDEX] != -1:
            alloc_idx = self._values[0, self.NEXT_INDEX]
            self._values[0, self.NEXT_INDEX] = self._values[alloc_idx, self.NEXT_INDEX]
            return alloc_idx - 1  # 返回实际索引
        return None

    def is_empty(self) -> bool:
        """Check if the linked list is empty."""
        return self._values[0, self.NEXT_INDEX] == -1

    def free(self, idx: int) -> None:
        """Free a previously allocated index."""
        assert 0 <= idx < self.size - 1, "Index out of bounds"
        self._values[idx + 1, self.NEXT_INDEX] = self._values[0, self.NEXT_INDEX]
        self._values[0, self.NEXT_INDEX] = idx + 1

    def test_is_full(self):
        count = 0
        idx = 0
        while self._values[idx, self.NEXT_INDEX] != -1:
            count += 1
            idx = self._values[idx, self.NEXT_INDEX]

        return count == self.size - 1
