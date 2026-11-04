import pytest
import time
import sys
import uuid
import weakref
from multiprocessing import Process
from lightllm.server.core.objs.atomic_array_lock import AtomicShmArrayLock, AtomicLockItem, AsyncLock
from lightllm.utils import shm_utils


@pytest.fixture(autouse=True)
def service_name(monkeypatch):
    name = f"test_atomic_array_{uuid.uuid4().hex}"
    monkeypatch.setattr(shm_utils, "get_unique_server_name", lambda: name)


@pytest.fixture
def setup_shared_memory():
    lock_name = "test_array_lock"
    lock_num = 2
    # Create an instance of AtomicShmArrayLock
    lock = AtomicShmArrayLock(lock_name, lock_num)
    yield lock
    lock.shm.unlink()  # Clean up the shared memory after tests


def test_initialization(setup_shared_memory):
    lock = setup_shared_memory
    assert lock.lock_num == 2
    assert lock.shm.size == 8  # 4 bytes per lock * 2 locks


def test_lock(setup_shared_memory):
    with setup_shared_memory.get_lock_context(0):
        assert setup_shared_memory.shm.buf.cast("i")[0] == 1
        assert setup_shared_memory.shm.buf.cast("i")[1] == 0
        with setup_shared_memory.get_lock_context(1):
            assert setup_shared_memory.shm.buf.cast("i")[0] == 1
            assert setup_shared_memory.shm.buf.cast("i")[1] == 1
    assert setup_shared_memory.shm.buf.cast("i")[0] == 0
    assert setup_shared_memory.shm.buf.cast("i")[1] == 0


def test_locking_and_unlocking(setup_shared_memory):
    lock = setup_shared_memory

    def worker(lock_index):
        with lock.get_lock_context(lock_index):
            time.sleep(0.1)  # Simulate some work while holding the lock

    # Start two processes trying to acquire locks
    process1 = Process(target=worker, args=(0,))
    process2 = Process(target=worker, args=(1,))

    process1.start()
    time.sleep(0.01)  # Ensure process1 acquires the lock first
    process2.start()

    process1.join()
    process2.join()

    # After both processes finish, check that the locks are released
    assert lock.shm.buf.cast("i")[0] == 0
    assert lock.shm.buf.cast("i")[1] == 0


def test_lock_exceeding_index(setup_shared_memory):
    lock = setup_shared_memory
    with pytest.raises(AssertionError):
        lock.get_lock_context(lock.lock_num)  # Should raise an error for invalid index


@pytest.mark.parametrize("asynchronous", [False, True])
def test_last_lock_item_releases_mapping_without_exported_views(asynchronous, monkeypatch):
    errors = []
    monkeypatch.setattr(sys, "unraisablehook", lambda error: errors.append(str(error.exc_value)))
    pool = AtomicShmArrayLock(f"lock_lifetime_{uuid.uuid4().hex}", 1)
    pool.shm.unlink()
    mapping = weakref.ref(pool.shm)
    item = pool.get_lock_context(0)
    wrapper = AsyncLock(item) if asynchronous else item
    del pool
    assert mapping() is not None
    assert item.try_acquire()
    assert not item.try_acquire()
    item.release()
    with pytest.raises(ValueError, match="request failed"), item:
        raise ValueError("request failed")
    assert item.try_acquire()
    item.release()
    del item, wrapper
    assert mapping() is None
    assert errors == []


if __name__ == "__main__":
    pytest.main()
