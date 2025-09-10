import os
import pickle
import numpy as np
from multiprocessing import shared_memory
from typing import List
from lightllm.utils.envs_utils import get_unique_server_name
from lightllm.utils.log_utils import init_logger
from lightllm.utils.shm_utils import create_or_link_shm

logger = init_logger(__name__)

LIGHTLLM_RPC_BYTE_SIZE = int(os.getenv("LIGHTLLM_RPC_BYTE_SIZE", 1024 * 1024 * 16))  # 默认16M buf
LIGHTLLM_RPC_RESULT_BYTE_SIZE = int(os.getenv("LIGHTLLM_RPC_RESULT_BYTE_SIZE", 1024 * 1024))  # 默认1M buf


class RpcShmParams:
    def __init__(self):
        self.shm = None
        self.name = f"{get_unique_server_name()}_RpcShmParams"

    def create_or_link_shm(self):
        self.shm = create_or_link_shm(self.name, LIGHTLLM_RPC_BYTE_SIZE)

        return

    def write_func_params(self, func_name, args):
        objs_bytes = pickle.dumps((func_name, args))
        self.shm.buf.cast("i")[0] = len(objs_bytes)
        self.shm.buf[4 : 4 + len(objs_bytes)] = objs_bytes
        return

    def read_func_params(self):
        bytes_len = self.shm.buf.cast("i")[0]
        func_name, args = pickle.loads(self.shm.buf[4 : 4 + bytes_len])
        return func_name, args


class RpcShmResults:
    def __init__(self):
        self.shm = None
        self.name = f"{get_unique_server_name()}_RpcShmResults"

    def create_or_link_shm(self):
        self.shm = create_or_link_shm(self.name, LIGHTLLM_RPC_RESULT_BYTE_SIZE)
        return

    def write_func_result(self, func_name, ret):
        objs_bytes = pickle.dumps((func_name, ret))
        self.shm.buf.cast("i")[0] = len(objs_bytes)
        self.shm.buf[4 : 4 + len(objs_bytes)] = objs_bytes

    def read_func_result(self):
        bytes_len = self.shm.buf.cast("i")[0]
        func_name, ret = pickle.loads(self.shm.buf[4 : 4 + bytes_len])
        return func_name, ret


class ShmSyncStatusArray:
    def __init__(self, world_size):
        self.shm = None
        self.arr = None
        self.name = f"{get_unique_server_name()}_rpc_result_state"
        self.dtype_byte_num = np.array([1], dtype=np.int64).dtype.itemsize
        self.dest_size = np.prod((world_size * 2,)) * self.dtype_byte_num
        self.shape = (world_size * 2,)
        self.dtype = np.int64
        self.world_size = world_size

    def create_or_link_shm(self):
        self.shm = create_or_link_shm(self.name, self.dest_size)

        self.arr = np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf)
        self.arr[:] = 0
        self.arr0 = self.arr[0 : self.world_size]
        self.arr1 = self.arr[self.world_size : 2 * self.world_size]
        return

    def add_mark(self, tp_rank: int):
        self.arr0[tp_rank] += 1
        return

    def add_mark1(self, tp_rank: int):
        self.arr1[tp_rank] += 1
        return

    def run_finished(self):
        return len(np.unique(self.arr0)) == 1

    def run_finished1(self):
        return len(np.unique(self.arr1)) == 1
