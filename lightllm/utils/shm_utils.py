from multiprocessing import shared_memory
import logging
import os
import fcntl

logger = logging.getLogger(__name__)

LIGHTLLM_SHM_LOCK_FILE = f"/tmp/lightllm_shm_creation_{os.getuid()}.lock"


def acquire_lock():
    lock_fd = os.open(LIGHTLLM_SHM_LOCK_FILE, os.O_CREAT | os.O_RDWR)
    fcntl.flock(lock_fd, fcntl.LOCK_EX)
    return lock_fd


def release_lock(lock_fd):
    fcntl.flock(lock_fd, fcntl.LOCK_UN)
    os.close(lock_fd)


def create_or_link_shm(name, expected_size):
    lock_fd = acquire_lock()
    # In case the size mismatch.
    safe_name = f"lightllm_{name}_{expected_size}"
    try:
        try:
            shm = shared_memory.SharedMemory(name=safe_name, create=True, size=expected_size)
            logger.info(f"Created new shared memory: {name} (size={expected_size})")
            return shm
        except FileExistsError:
            try:
                shm = shared_memory.SharedMemory(name=safe_name)
            except FileNotFoundError:
                logger.warning(f"Shared memory {name} disappeared, retrying...")
                shm = shared_memory.SharedMemory(name=safe_name, create=True, size=expected_size)
            except Exception as e:
                logger.error(f"Unexpected error attaching to shared memory {name}: {e}")
                raise
            return shm
    except Exception as e:
        logger.error(f"Unexpected error creating shared memory {safe_name}: {e}")
        raise
    finally:
        release_lock(lock_fd)
