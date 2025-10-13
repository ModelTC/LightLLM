import atomics
from multiprocessing import shared_memory
from lightllm.utils.log_utils import init_logger
from lightllm.utils.shm_utils import create_or_link_shm

logger = init_logger(__name__)


class AtomicShmLock:
    def __init__(self, lock_name: str):
        self.lock_name = lock_name
        self.dest_size = 4
        self.shm = create_or_link_shm(self.lock_name, self.dest_size)

        self.shm.buf.cast("i")[0] = 0
        return

    def __enter__(self):
        with atomics.atomicview(buffer=self.shm.buf, atype=atomics.INT) as a:
            while not a.cmpxchg_weak(0, 1):
                pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        with atomics.atomicview(buffer=self.shm.buf, atype=atomics.INT) as a:
            while not a.cmpxchg_weak(1, 0):
                pass
        return False
