import ctypes
import dataclasses
import xxhash
import numpy as np
from functools import lru_cache
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.log_utils import init_logger
from lightllm.utils.config_utils import get_config_json
from typing import List, Tuple, Optional

logger = init_logger(__name__)


def compute_token_list_hash(tokens: List[int], cpu_cache_token_page_size: int) -> List[int]:
    if len(tokens) == 0:
        return []

    chunks_hash_value = []
    hsum = xxhash.xxh3_64()

    # 计算每个分块的哈希值, 但是输入token需要少一个，因为
    # 如果计算所有的token，会导致输入input_len 命中全长的
    # cpu cache, 导致prefill 过程无法有输入来导出下一个输出。
    calcu_num = (len(tokens) - 1) // cpu_cache_token_page_size

    for i in range(calcu_num):
        start_index = i * cpu_cache_token_page_size
        end_index = (i + 1) * cpu_cache_token_page_size
        chunk = tokens[start_index:end_index]
        chunk_np = np.array(chunk, dtype=np.uint64)
        hsum.update(chunk_np.tobytes())

        hash_value = hsum.intdigest()
        chunks_hash_value.append(hash_value)

    return chunks_hash_value


@lru_cache(maxsize=None)
def calcu_cpu_cache_meta() -> "CpuKVCacheMeta":
    args = get_env_start_args()
    assert args.enable_cpu_cache
    model_config = get_config_json(args.model_dir)
    item_size = 2
    head_dim = model_config["hidden_size"] // model_config["num_attention_heads"]
    num_key_value_heads = model_config["num_key_value_heads"] * 2  # key and value
    layer_num = model_config["num_hidden_layers"]

    one_token_byte_size = layer_num * num_key_value_heads * head_dim * item_size
    one_page_byte_size = args.cpu_cache_token_page_size * one_token_byte_size
    cpu_cache_page_num = int((args.cpu_cache_storage_size * 1024 * 1024 * 1024) / one_page_byte_size)

    cpu_cache_meta = CpuKVCacheMeta(
        page_num=cpu_cache_page_num,
        layer_num=layer_num,
        token_page_size=args.cpu_cache_token_page_size,
        num_heads=num_key_value_heads,
        head_dim=head_dim,
        item_size=item_size,
    )

    logger.info(f"cpu kv cache page num: {cpu_cache_meta.page_num}")

    return cpu_cache_meta


@lru_cache(maxsize=None)
def create_shm_kv_cache_ptr() -> int:
    args = get_env_start_args()

    # 加载 libc
    libc = ctypes.CDLL("/usr/lib/x86_64-linux-gnu/libc.so.6")

    # 设置 shmget 函数的参数类型和返回类型
    libc.shmget.argtypes = (ctypes.c_long, ctypes.c_size_t, ctypes.c_int)
    libc.shmget.restype = ctypes.c_int

    # 设置 shmat 函数的参数类型和返回类型
    libc.shmat.argtypes = (ctypes.c_int, ctypes.c_void_p, ctypes.c_int)
    libc.shmat.restype = ctypes.c_void_p

    # 创建共享内存
    key = args.cpu_kv_cache_shm_id  # 共享内存的键
    size = calcu_cpu_cache_meta().calcu_size()  # 共享内存大小
    shmflg = 0o666 | 0o1000  # 权限和 IPC_CREAT 标志

    shmid = libc.shmget(key, size, shmflg)

    if shmid < 0:
        raise Exception("Error creating shared memory")

    logger.info(f"Shared memory ID: {shmid}")

    # 附加共享内存
    shm_addr = libc.shmat(shmid, ctypes.c_void_p(0), 0)

    if shm_addr == ctypes.c_void_p(-1).value:
        raise Exception("Error attaching shared memory")

    logger.info(f"Shared memory attached at address: {shm_addr}")

    # 自动注册清理
    try:
        from lightllm.utils.auto_shm_cleanup import auto_register_sysv_shm

        auto_register_sysv_shm(key, shmid)
        logger.info(f"Auto-registered shared memory key {key} (shmid {shmid}) for cleanup")
    except Exception as e:
        logger.warning(f"Failed to register auto shm cleanup: {e}")

    return shm_addr


@lru_cache(maxsize=None)
def attach_shm_kv_cache_ptr() -> int:
    """
    Attach to the shared memory segment with the given shmid.
    """
    args = get_env_start_args()
    libc = ctypes.CDLL("/usr/lib/x86_64-linux-gnu/libc.so.6")

    # 设置 shmget 和 shmat 函数的参数类型和返回类型
    libc.shmget.argtypes = (ctypes.c_long, ctypes.c_size_t, ctypes.c_int)
    libc.shmget.restype = ctypes.c_int
    libc.shmat.argtypes = (ctypes.c_int, ctypes.c_void_p, ctypes.c_int)
    libc.shmat.restype = ctypes.c_void_p

    # 通过键获取共享内存 ID
    key = args.cpu_kv_cache_shm_id  # 共享内存的键
    shmid = libc.shmget(key, 0, 0)

    if shmid < 0:
        raise Exception("Error getting shared memory")

    logger.info(f"Shared memory ID: {shmid}")

    # 附加共享内存
    shm_addr = libc.shmat(shmid, ctypes.c_void_p(0), 0)

    if shm_addr == ctypes.c_void_p(-1).value:
        raise Exception("Error attaching shared memory")

    logger.info(f"Shared memory attached at address: {shm_addr}")
    return shm_addr


@dataclasses.dataclass
class CpuKVCacheMeta:
    page_num: int
    layer_num: int
    token_page_size: int
    num_heads: int
    head_dim: int
    item_size: int

    def calcu_size(self):
        return self.page_num * self.layer_num * self.token_page_size * self.num_heads * self.head_dim * self.item_size


def register_shm_ptr_to_pin(shm_ptr: int, size: int) -> int:
    # 加载 CUDA 库
    cuda = ctypes.CDLL("/usr/local/cuda/targets/x86_64-linux/lib/libcudart.so")  # Linux 下的 CUDA 库路径

    # 定义 cudaHostRegister 函数的参数和返回类型
    cuda.cudaHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
    cuda.cudaHostRegister.restype = ctypes.c_int

    # 定义 cudaHostGetDevicePointer 函数原型
    cuda.cudaHostGetDevicePointer.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_int]
    cuda.cudaHostGetDevicePointer.restype = ctypes.c_int

    # 定义常量
    cudaHostRegisterDefault = 0  # 默认注册标志

    # 调用 cudaHostRegister
    result = cuda.cudaHostRegister(shm_ptr, size, cudaHostRegisterDefault)

    if result != 0:
        raise Exception(f"Error registering host memory: {result}")
    else:
        logger.info("Host memory registered successfully.")

    device_ptr = ctypes.c_void_p()  # 输出设备指针
    host_ptr = ctypes.c_void_p(shm_ptr)  # 输入主机指针

    result = cuda.cudaHostGetDevicePointer(ctypes.byref(device_ptr), host_ptr, 0)

    if result != 0:
        raise RuntimeError(f"cudaHostGetDevicePointer failed with error code {result}")

    logger.info(f"get Host memory registered Device ptr {device_ptr.value}")

    return device_ptr.value
