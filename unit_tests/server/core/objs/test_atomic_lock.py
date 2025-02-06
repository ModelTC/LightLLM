import pytest
import time
from multiprocessing import Process
from lightllm.utils.log_utils import init_logger
from lightllm.server.core.objs.atomic_lock import AtomicShmLock

logger = init_logger(__name__)


@pytest.fixture(scope="module")
def setup_shared_memory():
    lock_name = "test_lock"
    lock = AtomicShmLock(lock_name)
    yield lock
    lock.shm.unlink()  # Clean up after tests


def test_lock_acquisition(setup_shared_memory):
    lock = setup_shared_memory
    assert lock.shm.buf.cast("i")[0] == 0  # Initially, the lock should be free

    with lock:
        assert lock.shm.buf.cast("i")[0] == 1  # Lock should be acquired

    assert lock.shm.buf.cast("i")[0] == 0  # Lock should be released


def test_multiple_processes_locking():
    lock_name = "test_lock"
    lock = AtomicShmLock(lock_name)

    # Function to be run in a separate process
    def acquire_lock():
        with lock:
            time.sleep(1)  # Hold the lock for a second

    # Start a process that acquires the lock
    p = Process(target=acquire_lock)
    p.start()

    time.sleep(0.1)  # Ensure the first process has acquired the lock
    assert lock.shm.buf.cast("i")[0] == 1  # The lock should be held by the first process

    # Wait for the first process to finish
    p.join()
    assert lock.shm.buf.cast("i")[0] == 0  # The lock should be released after the process finishes


def test_lock_recreation_on_size_mismatch():
    lock_name = "test_lock_size_mismatch"
    lock = AtomicShmLock(lock_name)

    # Simulate a size mismatch by unlinking the shared memory
    lock.shm.unlink()

    # Now recreate the lock
    lock_new = AtomicShmLock(lock_name)
    assert lock_new.shm.size == lock_new.dest_size  # Ensure the size is correct
    assert lock_new.shm.buf.cast("i")[0] == 0  # Ensure the lock is free

    lock_new.shm.unlink()  # Clean up after tests


# 运行测试
if __name__ == "__main__":
    pytest.main()
