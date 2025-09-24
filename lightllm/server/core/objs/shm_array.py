import numpy as np
from multiprocessing import shared_memory
from lightllm.utils.log_utils import init_logger
from lightllm.utils.shm_utils import create_or_link_shm

logger = init_logger(__name__)


class ShmArray:
    def __init__(self, name, shape, dtype):
        self.shm = None
        self.arr = None
        self.name = name
        self.dtype_byte_num = np.array([1], dtype=dtype).dtype.itemsize
        self.dest_size = np.prod(shape) * self.dtype_byte_num
        self.shape = shape
        self.dtype = dtype

    def create_shm(self):
        self.shm = create_or_link_shm(self.name, self.dest_size)
        self.arr = np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf)

    def link_shm(self):
        self.shm = create_or_link_shm(self.name, self.dest_size, force_mode="link")
        assert self.shm.size == self.dest_size
        self.arr = np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf)
        return

    def close_shm(self):
        if self.shm is not None:
            self.shm.close()
            self.shm.unlink()
            self.shm = None
            self.arr = None
