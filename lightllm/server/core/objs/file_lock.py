"""Darwin process locks for the existing shared request buffers.

The atomics wheel bundles an x86-only macOS library. FileLock provides both
thread ownership and cross-process exclusion without loading that library.
"""
from threading import Lock
from filelock import FileLock, Timeout
from lightllm.utils.shm_utils import get_service_shm_name


class FileShmLock:
    def __init__(self, lock_name):
        self.lock_name = lock_name
        self._gate = Lock()
        self._lock = FileLock(f"/tmp/{get_service_shm_name(lock_name)}.process_lock")

    def __enter__(self):
        self._gate.acquire()
        try:
            self._lock.acquire()
        except BaseException:
            self._gate.release()
            raise
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False

    def try_acquire(self):
        if not self._gate.acquire(blocking=False):
            return False
        try:
            self._lock.acquire(timeout=0)
            return True
        except Timeout:
            self._gate.release()
            return False
        except BaseException:
            self._gate.release()
            raise

    def acquire_sleep1ms(self):
        self.__enter__()

    def release(self):
        self._lock.release()
        self._gate.release()


class FileShmArrayLock:
    def __init__(self, lock_name, lock_num):
        self.lock_name = lock_name
        self.lock_num = lock_num
        # Retain one lock object per slot per process so ownership is tracked
        # consistently across synchronous and asynchronous callers.
        self._locks = [FileShmLock(f"{lock_name}_{index}") for index in range(lock_num)]

    def get_lock_context(self, lock_index):
        if not 0 <= lock_index < self.lock_num:
            raise IndexError(lock_index)
        return self._locks[lock_index]
