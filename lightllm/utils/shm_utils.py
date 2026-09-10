from multiprocessing import shared_memory
from filelock import FileLock
from lightllm.utils.envs_utils import get_unique_server_name
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


def get_service_shm_name(name):
    """为内部共享内存统一添加当前服务（UUID + node rank）前缀。

    已带当前服务前缀的完整名称保持不变，便于底层包装函数安全复用。service name
    未初始化时直接报错，避免创建无法区分服务、也无法被 launcher 定向回收的裸名称。
    """
    name = str(name)
    service_name = get_unique_server_name()
    if not service_name:
        raise RuntimeError(
            "LIGHTLLM_UNIQUE_SERVICE_NAME_ID is unset; " "call set_unique_server_name(args) before using shared memory"
        )
    prefix = f"{service_name}_"
    return name if name.startswith(prefix) else f"{prefix}{name}"


def create_or_link_shm(name, expected_size, force_mode=None):
    """
    Args:
        name: logical name of the shared memory; the current service prefix is added here
        expected_size: expected size of the shared memory, if expected_size == -1, no check for size linked.
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
    name = get_service_shm_name(name)
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
    """强制连接到已存在的共享内存,
    如果 expected_size 为 -1, 则不进行link的size校验比对"""
    try:
        shm = shared_memory.SharedMemory(name=name)
        # 验证大小
        if expected_size != -1 and shm.size != expected_size:
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
