import atomics
import time
from multiprocessing import shared_memory
from lightllm.utils.log_utils import init_logger
from lightllm.utils.shm_utils import create_or_link_shm

logger = init_logger(__name__)


class AtomicShmLock:
    def __init__(self, lock_name: str):
        self.lock_name = lock_name
        self.dest_size = 4
        self.shm = create_or_link_shm(self.lock_name, self.dest_size)

        # New shared memory is zero-filled; attaching must preserve a held lock.
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

    # acquire_sleep1ms 和 release 是某些特定场景下主动使用进行锁获取的操作函数
    def acquire_sleep1ms(self):
        start_time = time.monotonic()
        retry_delay = 0.00005
        warned = False
        with atomics.atomicview(buffer=self.shm.buf, atype=atomics.INT) as a:
            while not a.cmpxchg_weak(0, 1):
                if not warned and time.monotonic() - start_time > 0.010:
                    logger.warning("acquire_sleep1ms wait more than 10ms")
                    warned = True
                # Long contention still backs off to the original 1ms interval.
                time.sleep(retry_delay)
                retry_delay = min(0.001, retry_delay * 2)

    def release(self):
        with atomics.atomicview(buffer=self.shm.buf, atype=atomics.INT) as a:
            while not a.cmpxchg_weak(1, 0):
                pass
        return
