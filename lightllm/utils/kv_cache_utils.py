import torch
import ctypes
import dataclasses
import os
import xxhash
import threading
import time
import numpy as np
import triton
from functools import lru_cache
from lightllm.utils.envs_utils import get_env_start_args, enable_huge_page, get_llm_data_type
from lightllm.utils.log_utils import init_logger
from lightllm.utils.config_utils import get_num_key_value_heads, get_head_dim, get_layer_num
from lightllm.common.kv_cache_mem_manager.mem_utils import select_mem_manager_class
from lightllm.common.kv_cache_mem_manager import (
    MemoryManager,
    INT8KVMemoryManager,
    CalibrationFP8KVMemoryManager,
    ExportCalibrationMemoryManager,
    PPLINT8KVMemoryManager,
    PPLINT4KVMemoryManager,
    Deepseek2MemoryManager,
    Deepseek2FP8KVMemoryManager,
)

from typing import List, Tuple, Optional
from tqdm import tqdm
from lightllm.utils.auto_shm_cleanup import register_sysv_shm_for_cleanup
from lightllm.utils.dist_utils import get_current_device_id

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

    mem_manager_class = select_mem_manager_class()
    if mem_manager_class is Deepseek2MemoryManager:
        cpu_cache_meta = CpuKVCacheMeta(
            page_num=0,
            token_page_size=args.cpu_cache_token_page_size,
            layer_num=get_layer_num(args.model_dir),
            num_heads=1,
            head_dim=512 + 64,
            data_type=get_llm_data_type(),
            scale_head_dim=0,
            scale_data_type=get_llm_data_type(),
        )
    elif mem_manager_class is Deepseek2FP8KVMemoryManager:
        cpu_cache_meta = CpuKVCacheMeta(
            page_num=0,
            token_page_size=args.cpu_cache_token_page_size,
            layer_num=get_layer_num(args.model_dir),
            num_heads=1,
            head_dim=512 + 64 + 2,
            data_type=torch.uint8,
            scale_head_dim=0,
            scale_data_type=get_llm_data_type(),
        )
    elif mem_manager_class is MemoryManager:
        cpu_cache_meta = CpuKVCacheMeta(
            page_num=0,
            token_page_size=args.cpu_cache_token_page_size,
            layer_num=get_layer_num(args.model_dir),
            num_heads=get_num_key_value_heads(args.model_dir) * 2,
            head_dim=get_head_dim(args.model_dir),
            data_type=get_llm_data_type(),
            scale_head_dim=0,
            scale_data_type=get_llm_data_type(),
        )
    elif mem_manager_class is PPLINT8KVMemoryManager:
        cpu_cache_meta = CpuKVCacheMeta(
            page_num=0,
            token_page_size=args.cpu_cache_token_page_size,
            layer_num=get_layer_num(args.model_dir),
            num_heads=get_num_key_value_heads(args.model_dir) * 2,
            head_dim=get_head_dim(args.model_dir),
            data_type=torch.int8,
            scale_head_dim=get_head_dim(args.model_dir) // 8,
            scale_data_type=get_llm_data_type(),
        )
    else:
        logger.error(f"not support mem manager: {mem_manager_class} for cpu kv cache")
        raise Exception(f"not support mem manager: {mem_manager_class} for cpu kv cache")

    if args.mtp_mode is not None:
        # TODO 可能会存在不同mtp模式的精度问题
        cpu_cache_meta.layer_num += 1

    cpu_cache_page_num = int(
        (args.cpu_cache_storage_size * 1024 * 1024 * 1024) / (cpu_cache_meta.calcu_one_page_size())
    )
    cpu_cache_meta.page_num = cpu_cache_page_num

    logger.info(f"cpu kv cache page num: {cpu_cache_meta.page_num}")

    return cpu_cache_meta


@lru_cache(maxsize=None)
def create_shm_kv_cache_ptr() -> int:
    libc = ctypes.CDLL("/usr/lib/x86_64-linux-gnu/libc.so.6", use_errno=True)
    libc.shmget.argtypes = (ctypes.c_long, ctypes.c_size_t, ctypes.c_int)
    libc.shmget.restype = ctypes.c_int
    libc.shmat.argtypes = (ctypes.c_int, ctypes.c_void_p, ctypes.c_int)
    libc.shmat.restype = ctypes.c_void_p

    args = get_env_start_args()
    key = args.cpu_kv_cache_shm_id
    requested_size = calcu_cpu_cache_meta().calcu_size()
    use_hugetlb = enable_huge_page()

    # 计算大页大小（默认从 /proc/meminfo 读取 Hugepagesize）
    def _get_default_hugepage_size() -> int:
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("Hugepagesize:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            kb = int(parts[1])
                            return kb * 1024
        except Exception:
            pass
        return 2 * 1024 * 1024  # fallback 2MB

    shmflg = 0o666 | 0o1000  # 权限和 IPC_CREAT 标志
    if use_hugetlb:
        # 向上对齐到大页大小
        huge_sz = _get_default_hugepage_size()
        size_to_alloc = triton.cdiv(requested_size, huge_sz) * huge_sz
        SHM_HUGETLB = 0o4000
        shmflg |= SHM_HUGETLB
        logger.info(
            f"Using SHM_HUGETLB, hugepage_size={huge_sz} bytes, requested={requested_size}, alloc={size_to_alloc}"
        )
    else:
        size_to_alloc = requested_size
        logger.info(f"Using regular pages, requested={requested_size}, alloc={size_to_alloc}")

    shmid = libc.shmget(key, size_to_alloc, shmflg)
    hugepages_num = (size_to_alloc + 1024 * 1024 * 1024 - 1) // (1024 * 1024 * 1024)
    if shmid < 0:
        err = ctypes.get_errno()
        if use_hugetlb:
            raise Exception(
                f"shmget with SHM_HUGETLB failed (errno={err}). Falling back to regular pages."
                f"You may need to configure hugepages manually, e.g.,"
                f"sudo sed -i 's/^GRUB_CMDLINE_LINUX=\"/& default_hugepagesz=1G \
                    hugepagesz=1G hugepages={hugepages_num}/' /etc/default/grub"
                f"sudo update-grub"
                f"sudo reboot"
            )
        else:
            raise Exception(f"Error creating regular shared memory (errno={err})")

    register_sysv_shm_for_cleanup(key, shmid)
    logger.info(f"Shared memory ID: {shmid}")

    # 附加共享内存
    shm_addr = libc.shmat(shmid, ctypes.c_void_p(0), 0)
    if shm_addr == ctypes.c_void_p(-1).value:
        raise Exception("Error attaching shared memory")
    logger.info(f"Shared cpu kv cache tensor memory at address: {shm_addr}")

    return shm_addr


@dataclasses.dataclass
class CpuKVCacheMeta:
    page_num: int
    token_page_size: int
    layer_num: int
    num_heads: int
    head_dim: int
    data_type: torch.dtype
    scale_head_dim: int
    scale_data_type: torch.dtype

    def calcu_size(self):
        return self.page_num * self.calcu_one_page_size()

    def calcu_one_page_size(self):
        return (
            self.token_page_size
            * self.layer_num
            * self.num_heads
            * (self.head_dim * self.data_type.itemsize + self.scale_head_dim * self.scale_data_type.itemsize)
        )

    def get_merged_head_dim(self):
        """
        返回将head_dim 和 scale_head_dim 看成融合成一个head_dim时候, head_dim的长度。
        """
        assert (
            self.head_dim * self.data_type.itemsize + self.scale_head_dim * self.scale_data_type.itemsize
        ) % self.data_type.itemsize == 0
        return (
            self.head_dim * self.data_type.itemsize + self.scale_head_dim * self.scale_data_type.itemsize
        ) // self.data_type.itemsize


@lru_cache(maxsize=None)
def register_shm_ptr_to_pin(shm_ptr: int, size: int) -> "AsyncRegistrationHandle":
    """Start async cudaHostRegister on the given [shm_ptr, shm_ptr+size) and return a handle."""
    chunk_bytes = 128 * 1024 * 1024  # 128M性能最好
    tasks: list[tuple[int, int]] = []
    offset = 0
    while offset < size:
        seg_len = min(chunk_bytes, size - offset)
        tasks.append((offset, seg_len))
        offset += seg_len

    handle = AsyncRegistrationHandle(total_tasks=len(tasks))

    def _worker():
        cuda = ctypes.CDLL("/usr/local/cuda/targets/x86_64-linux/lib/libcudart.so")
        cuda.cudaHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
        cuda.cudaHostRegister.restype = ctypes.c_int
        cuda.cudaHostGetDevicePointer.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_int]
        cuda.cudaHostGetDevicePointer.restype = ctypes.c_int

        cudaHostRegisterFlag = 3

        torch.cuda.set_device(get_current_device_id())
        # TODO 这个地方的分块注册是否具备合法性和合理性。
        for offset, seg_len in tasks:
            ptr = ctypes.c_void_p(shm_ptr + offset)
            r = cuda.cudaHostRegister(ptr, ctypes.c_size_t(seg_len), cudaHostRegisterFlag)
            if r != 0:
                raise Exception(f"cudaHostRegister failed with error code {r}, prefer to use hugetlb")
            handle.task_count += 1

        device_ptr = ctypes.c_void_p()
        host_ptr = ctypes.c_void_p(shm_ptr)
        res = cuda.cudaHostGetDevicePointer(ctypes.byref(device_ptr), host_ptr, 0)
        if res != 0:
            raise Exception(f"cudaHostGetDevicePointer failed with error code {res}")
        assert host_ptr.value == device_ptr.value
        handle.tasks_finished.set()

    th = threading.Thread(target=_worker, name="cpu_cache_register", daemon=True)
    handle.thread = th
    th.start()
    return handle


class AsyncRegistrationHandle:
    """A handle for async host memory registration.

    - wait(): blocks until registration finishes, prints tqdm progress, and returns device pointer (int).
    """

    def __init__(self, total_tasks: int):
        self.total_tasks = total_tasks
        self.task_count = 0
        self.thread: Optional[threading.Thread] = None
        self.tasks_finished = threading.Event()

    def wait(self):
        """Block until the async registration completes. Only here we print tqdm progress."""
        last_count = 0
        desc = f"pid {os.getpid()} Registering pinned host memory (async)"
        with tqdm(total=self.total_tasks, desc=desc) as pbar:
            while not self.tasks_finished.is_set():
                cur = self.task_count
                if cur > last_count:
                    pbar.update(cur - last_count)
                    last_count = cur
                time.sleep(0.01)
            # final update
            cur = self.task_count
            if cur > last_count:
                pbar.update(cur - last_count)
                last_count = cur

        if self.thread is not None and self.thread.is_alive():
            self.thread.join()

        return


@lru_cache(maxsize=None)
def attach_shm_kv_cache_ptr() -> int:
    libc = ctypes.CDLL("/usr/lib/x86_64-linux-gnu/libc.so.6", use_errno=True)
    libc.shmget.argtypes = (ctypes.c_long, ctypes.c_size_t, ctypes.c_int)
    libc.shmget.restype = ctypes.c_int
    libc.shmat.argtypes = (ctypes.c_int, ctypes.c_void_p, ctypes.c_int)
    libc.shmat.restype = ctypes.c_void_p

    # Try to locate an existing SHM without creating a new one
    args = get_env_start_args()
    key = args.cpu_kv_cache_shm_id
    shmid = libc.shmget(key, 0, 0)
    if shmid < 0:
        size = calcu_cpu_cache_meta().calcu_size()
        shmid = libc.shmget(key, size, 0)
    if shmid < 0:
        err = ctypes.get_errno()
        raise Exception(f"Error locating existing shared memory (errno={err})")

    shm_addr = libc.shmat(shmid, ctypes.c_void_p(0), 0)
    if shm_addr == ctypes.c_void_p(-1).value:
        err = ctypes.get_errno()
        raise Exception(f"Error attaching shared memory (errno={err})")

    logger.info(f"Attached to SHM key={key}, shmid={shmid}, addr={shm_addr}")
    return shm_addr
