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

        self.shm.buf.cast("i")[0] = 0
        self.acquire_time = None
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
        last_log_time = time.monotonic()
        with atomics.atomicview(buffer=self.shm.buf, atype=atomics.INT) as a:
            while not a.cmpxchg_weak(0, 1):
                now = time.monotonic()
                if now - last_log_time >= 0.1:
                    logger.warning("acquire_sleep1ms wait for 100ms")
                    last_log_time = now
                time.sleep(0.001)
                pass
        self.acquire_time = time.monotonic()

    def release(self, log_timeout=False, log_tag=""):
        if log_timeout and self.acquire_time is not None:
            hold_time = (time.monotonic() - self.acquire_time) * 1000  # 转换为毫秒
            if hold_time > 5:
                tag_str = f"[{log_tag}] " if log_tag else ""
                logger.warning(f"{tag_str}Lock {self.lock_name} held for {hold_time:.2f}ms (>5ms)")
            self.acquire_time = None
        with atomics.atomicview(buffer=self.shm.buf, atype=atomics.INT) as a:
            while not a.cmpxchg_weak(1, 0):
                pass
        return
