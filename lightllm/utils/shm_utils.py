from multiprocessing import shared_memory
import logging
import os
import fcntl

logger = logging.getLogger(__name__)

LIGHTLLM_SHM_LOCK_FILE = "/tmp/lightllm_shm_creation.lock"


def acquire_lock():
    lock_fd = os.open(LIGHTLLM_SHM_LOCK_FILE, os.O_CREAT | os.O_RDWR)
    fcntl.flock(lock_fd, fcntl.LOCK_EX)
    return lock_fd


def release_lock(lock_fd):
    fcntl.flock(lock_fd, fcntl.LOCK_UN)
    os.close(lock_fd)


def create_or_link_shm(name, expected_size):
    lock_fd = acquire_lock()
    try:
        try:
            shm = shared_memory.SharedMemory(name=name, create=True, size=expected_size)
            logger.info(f"Created new shared memory: {name} ({expected_size=})")
            return shm
        except FileExistsError:
            try:
                shm = shared_memory.SharedMemory(name=name)
            except FileNotFoundError:
                logger.warning(f"Shared memory {name} disappeared, retrying...")
                shm = shared_memory.SharedMemory(name=name, create=True, size=expected_size)
            except Exception as e:
                logger.error(f"Unexpected error attaching to shared memory {name}: {e}")
                raise
            if shm.size != expected_size:
                logger.warning(f"Size mismatch: expected {expected_size}, got {shm.size}. Recreating {name}...")
                shm.close()
                try:
                    shm.unlink()
                except FileNotFoundError:
                    pass
                shm = shared_memory.SharedMemory(name=name, create=True, size=expected_size)
            logger.info(f"Attached to existing shared memory: {name} ({expected_size=})")
            return shm
    except Exception as e:
        logger.error(f"Unexpected error creating shared memory {name}: {e}")
        raise
    finally:
        release_lock(lock_fd)
