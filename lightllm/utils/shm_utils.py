import time
from multiprocessing import shared_memory
import logging
import random

logger = logging.getLogger(__name__)

MAX_RETRIES = 10
RETRY_DELAY = 0.1  # seconds


def create_or_link_shm(name: str, expected_size: int) -> shared_memory.SharedMemory:
    for _ in range(MAX_RETRIES):
        shm = None
        try:
            shm = shared_memory.SharedMemory(name=name, create=True, size=expected_size)
            logger.info(f"Created new shared memory: {name} (size={expected_size})")
            return shm
        except FileExistsError:
            try:
                shm = shared_memory.SharedMemory(name=name, create=False)
            except FileNotFoundError:
                logger.warning(f"Shared memory {name} disappeared, retrying...")
                time.sleep(RETRY_DELAY * random.uniform(1, 2))
                continue

            if shm.size != expected_size:
                logger.warning(f"Size mismatch: expected {expected_size}, got {shm.size}. Recreating {name}...")
                shm.close()
                try:
                    shm.unlink()
                except FileNotFoundError:
                    pass

                time.sleep(RETRY_DELAY * random.uniform(1, 2))
                continue
            else:
                logger.info(f"Attached to existing shared memory: {name} (size={shm.size})")
                return shm
        except Exception as e:
            if shm:
                shm.close()
            logger.error(f"Unexpected error creating/attaching shm {name}: {e}")
            raise

    raise RuntimeError(f"Failed to create or attach to shared memory '{name}' after {MAX_RETRIES} attempts")
