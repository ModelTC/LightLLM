# import faulthandler
# faulthandler.enable()

import numpy as np
from multiprocessing import shared_memory
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class SharedArray:
    def __init__(self, name, shape, dtype):
        dtype_byte_num = np.array([1], dtype=dtype).dtype.itemsize
        dest_size = np.prod(shape) * dtype_byte_num
        try:
            shm = shared_memory.SharedMemory(name=name, create=True, size=dest_size)
            logger.info(f"create shm {name}")
        except:
            shm = shared_memory.SharedMemory(name=name, create=False, size=dest_size)
            logger.info(f"link shm {name}")

        if shm.size != dest_size:
            logger.info(f"size not same, unlink shm {name} and create again")
            shm.unlink()
            shm.close()
            try:
                shm = shared_memory.SharedMemory(name=name, create=True, size=dest_size)
                logger.info(f"create shm {name}")
            except Exception as e:
                shm = shared_memory.SharedMemory(name=name, create=False, size=dest_size)
                logger.info(f"error {str(e)} to link shm {name}")

        self.shm = shm  # SharedMemory 对象一定要被持有，否则会被释放
        self.arr = np.ndarray(shape, dtype=dtype, buffer=self.shm.buf)


class SharedInt(SharedArray):
    def __init__(self, name):
        super().__init__(name, shape=(1,), dtype=np.int64)

    def set_value(self, value):
        self.arr[0] = value

    def get_value(self):
        return self.arr[0]


if __name__ == "__main__":
    # test SharedArray
    a = SharedArray("sb_abc", (1,), dtype=np.int32)
    a.arr[0] = 10
    assert a.arr[0] == 10
    a.arr[0] += 10
    assert a.arr[0] == 20
