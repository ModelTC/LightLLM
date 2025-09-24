from multiprocessing import shared_memory
from filelock import FileLock
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


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
    lock_name = f"/tmp/{name}.lock"

    if force_mode == "create":
        with FileLock(lock_name):
            return _force_create_shm(name, expected_size)
    elif force_mode == "link":
        return _force_link_shm(name, expected_size)
    else:
        with FileLock(lock_name):
            return _smart_create_or_link_shm(name, expected_size)


def _force_create_shm(name, expected_size):
    """强制创建新的共享内存"""
    try:
        existing_shm = shared_memory.SharedMemory(name=name)
        existing_shm.close()
        existing_shm.unlink()
    except:
        pass

    # 创建新的共享内存
    shm = shared_memory.SharedMemory(name=name, create=True, size=expected_size)
    return shm


def _force_link_shm(name, expected_size):
    """强制连接到已存在的共享内存"""
    try:
        shm = shared_memory.SharedMemory(name=name)
        # 验证大小
        if shm.size != expected_size:
            shm.close()
            raise ValueError(f"Shared memory {name} size mismatch: expected {expected_size}, got {shm.size}")
        # logger.info(f"Force linked to existing shared memory: {name} (size={expected_size})")
        return shm
    except Exception as e:
        raise e


def _smart_create_or_link_shm(name, expected_size):
    """优先连接，不存在则创建"""
    try:
        shm = _force_link_shm(name=name, expected_size=expected_size)
        return shm
    except:
        pass

    return _force_create_shm(name=name, expected_size=expected_size)
