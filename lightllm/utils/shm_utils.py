from multiprocessing import shared_memory
import logging
import os
from filelock import FileLock

logger = logging.getLogger(__name__)

LIGHTLLM_SHM_LOCK_FILE = f"/tmp/lightllm_shm_creation_{os.getuid()}.lock"


def create_or_link_shm(name, expected_size, force_mode=None):
    """
    Args:
        name: name of the shared memory
        expected_size: expected size of the shared memory
        force_mode: force mode
            - 'create': force create new shared memory, if exists, delete and create
            - 'link': force link to existing shared memory, if not exists, raise exception
            - None (default): smart mode, link to existing, if not exists, create

    Returns:
        shared_memory.SharedMemory: shared memory object

    Raises:
        FileNotFoundError: when force_mode='link' but shared memory not exists
        ValueError: when force_mode='link' but size mismatch
    """
    lock = FileLock(LIGHTLLM_SHM_LOCK_FILE)
    safe_name = f"lightllm_{name}_{expected_size}"

    with lock:
        if force_mode == "create":
            return _force_create_shm(safe_name, name, expected_size)
        elif force_mode == "link":
            return _force_link_shm(safe_name, name, expected_size)
        else:
            return _smart_create_or_link_shm(safe_name, name, expected_size)


def _force_create_shm(safe_name, name, expected_size):
    """强制创建新的共享内存"""
    try:
        existing_shm = shared_memory.SharedMemory(name=safe_name)
        existing_shm.close()
        existing_shm.unlink()
        logger.info(f"Removed existing shared memory and force create: {safe_name}")
    except FileNotFoundError:
        pass  #

    # 创建新的共享内存
    shm = shared_memory.SharedMemory(name=safe_name, create=True, size=expected_size)
    logger.info(f"Force created new shared memory: {name} (size={expected_size})")
    return shm


def _force_link_shm(safe_name, name, expected_size):
    """强制连接到已存在的共享内存"""
    try:
        shm = shared_memory.SharedMemory(name=safe_name)
        # 验证大小
        if shm.size != expected_size:
            shm.close()
            raise ValueError(f"Shared memory {name} size mismatch: expected {expected_size}, got {shm.size}")
        # logger.info(f"Force linked to existing shared memory: {name} (size={expected_size})")
        return shm
    except FileNotFoundError:
        raise FileNotFoundError(f"Shared memory {name} not found for force link")


def _smart_create_or_link_shm(safe_name, name, expected_size):
    """优先连接，不存在则创建"""
    try:
        shm = shared_memory.SharedMemory(name=safe_name, create=True, size=expected_size)
        logger.info(f"Created new shared memory: {name} (size={expected_size})")
        return shm
    except FileExistsError:
        try:
            shm = shared_memory.SharedMemory(name=safe_name)
            # 验证共享内存大小是否匹配
            if shm.size != expected_size:
                logger.error(f"Shared memory {name} size mismatch: expected {expected_size}, got {shm.size}")
                shm.close()
                try:
                    shm.unlink()
                    logger.info(f"Removed mismatched shared memory: {safe_name}")
                except FileNotFoundError:
                    pass  # 已经被其他进程删除了
                shm = shared_memory.SharedMemory(name=safe_name, create=True, size=expected_size)
                logger.info(f"Recreated shared memory: {name} (size={expected_size})")
            else:
                logger.info(f"Linked to existing shared memory: {name} (size={expected_size})")
            return shm
        except FileNotFoundError:
            logger.warning(f"Shared memory {name} disappeared, retrying...")
            shm = shared_memory.SharedMemory(name=safe_name, create=True, size=expected_size)
            logger.info(f"Created new shared memory after disappearance: {name} (size={expected_size})")
            return shm
        except Exception as e:
            logger.error(f"Unexpected error attaching to shared memory {name}: {e}")
            raise
