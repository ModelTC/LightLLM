import torch
import ctypes
import dataclasses
import os
import xxhash
import threading
import concurrent.futures
import numpy as np
import triton
from functools import lru_cache
from lightllm.utils.envs_utils import (
    get_env_start_args,
    enable_huge_page,
    enable_cpu_cache_numa_interleave,
    get_llm_data_type,
    get_added_mtp_kv_layer_num,
)
from lightllm.utils.log_utils import init_logger
from lightllm.utils.config_utils import (
    get_config_json,
    get_num_key_value_heads,
    get_head_dim,
    get_layer_num,
    is_linear_att_mixed_model,
)
from lightllm.common.kv_cache_mem_manager.mem_utils import select_mem_manager_class
from lightllm.common.kv_cache_mem_manager import (
    MemoryManager,
    PPLINT8KVMemoryManager,
    PPLINT4KVMemoryManager,
    Deepseek2MemoryManager,
    Deepseek3_2MemoryManager,
    FP8PerTokenGroupQuantDeepseek3_2MemoryManager,
    DeepseekV4MemoryManager,
    Qwen3NextMemManager,
)

from typing import List, Tuple, Optional
from tqdm import tqdm
from lightllm.utils.auto_shm_cleanup import register_sysv_shm_for_cleanup
from lightllm.utils.dist_utils import get_current_device_id
from lightllm.common.linear_att_cache_manager.config_objs import LinearAttCacheConfig

logger = init_logger(__name__)


def compute_token_list_hash(tokens: List[int], cpu_cache_token_page_size: int) -> List[int]:
    if len(tokens) == 0:
        return []

    chunks_hash_value = []
    hsum = xxhash.xxh3_128()

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

    if is_linear_att_mixed_model(args.model_dir):
        # 对于 qwen3.5 等 linear att 混合模型的特殊处理。
        mem_manager_class = Qwen3NextMemManager
    else:
        mem_manager_class = select_mem_manager_class()

    if mem_manager_class is Qwen3NextMemManager:
        linear_config = LinearAttCacheConfig.load_from_args()
        cpu_cache_meta = CpuKVCacheMeta(
            page_num=0,
            token_page_size=1,
            layer_num=1,
            num_heads=1,
            head_dim=linear_config.get_cpu_cache_big_page_bytes(),
            data_type=torch.uint8,
            scale_head_dim=0,
            scale_data_type=get_llm_data_type(),
        )
    elif mem_manager_class is Deepseek2MemoryManager:
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
    elif mem_manager_class is Deepseek3_2MemoryManager:
        cpu_cache_meta = CpuKVCacheMeta(
            page_num=0,
            token_page_size=args.cpu_cache_token_page_size,
            layer_num=get_layer_num(args.model_dir),
            num_heads=1,
            head_dim=512 + 64 + (144 // 2),
            data_type=get_llm_data_type(),
            scale_head_dim=0,
            scale_data_type=get_llm_data_type(),
        )
    elif mem_manager_class is FP8PerTokenGroupQuantDeepseek3_2MemoryManager:
        cpu_cache_meta = CpuKVCacheMeta(
            page_num=0,
            token_page_size=args.cpu_cache_token_page_size,
            layer_num=get_layer_num(args.model_dir),
            num_heads=1,
            head_dim=(
                FP8PerTokenGroupQuantDeepseek3_2MemoryManager.flashmla_bytes_per_token
                + FP8PerTokenGroupQuantDeepseek3_2MemoryManager.indexer_bytes_per_token
            ),
            data_type=torch.uint8,
            scale_head_dim=0,
            scale_data_type=torch.uint8,
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
    elif mem_manager_class is DeepseekV4MemoryManager:
        from lightllm.common.kv_cache_mem_manager.deepseek4_mem_manager import DeepseekV4CpuCacheLayout

        config = get_config_json(args.model_dir)
        layer_num = get_layer_num(args.model_dir) + get_added_mtp_kv_layer_num()
        layout = DeepseekV4CpuCacheLayout.from_compress_rates(
            compress_rates=config["compress_ratios"][:layer_num],
            token_page_size=args.cpu_cache_token_page_size,
            head_dim=get_head_dim(args.model_dir),
            indexer_head_dim=config["index_head_dim"],
        )
        cpu_cache_meta = CpuKVCacheMeta(
            page_num=0,
            token_page_size=args.cpu_cache_token_page_size,
            layer_num=layer_num,
            num_heads=1,
            head_dim=layout.page_nbytes,
            data_type=torch.uint8,
            scale_head_dim=0,
            scale_data_type=torch.uint8,
            page_shape=(layout.page_nbytes,),
        )
    else:
        logger.error(f"not support mem manager: {mem_manager_class} for cpu kv cache")
        raise Exception(f"not support mem manager: {mem_manager_class} for cpu kv cache")

    if args.mtp_mode is not None and mem_manager_class is not DeepseekV4MemoryManager:
        # TODO 可能会存在不同mtp模式的精度问题
        if not is_linear_att_mixed_model(args.model_dir):
            # 对于非 linear att 混合模型，需要额外增加 mtp 的 kv 层数，
            # 对于 linear att 混合模型，如qwen 3.5 mtp，已经将 kv 数据
            # 打包成一个块了，所以不需要额外增加，其 layer_num 一直都保持为 1
            cpu_cache_meta.layer_num += get_added_mtp_kv_layer_num()

    cpu_cache_page_num = int(
        (args.cpu_cache_storage_size * 1024 * 1024 * 1024) / (cpu_cache_meta.calcu_one_page_size())
    )
    cpu_cache_meta.page_num = cpu_cache_page_num

    logger.info(f"cpu kv cache page num: {cpu_cache_meta.page_num}")

    return cpu_cache_meta


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
    page_shape: Optional[Tuple[int, ...]] = None

    def calcu_size(self):
        return self.page_num * self.calcu_one_page_size()

    def calcu_one_page_size(self):
        if self.page_shape is not None:
            return int(np.prod(self.page_shape)) * self.data_type.itemsize
        return (
            self.token_page_size
            * self.layer_num
            * self.num_heads
            * (self.head_dim * self.data_type.itemsize + self.scale_head_dim * self.scale_data_type.itemsize)
        )

    def get_tensor_shape(self) -> Tuple[int, ...]:
        if self.page_shape is not None:
            return (self.page_num, *self.page_shape)
        return (
            self.page_num,
            self.layer_num,
            self.token_page_size,
            self.num_heads,
            self.get_merged_head_dim(),
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
def create_shm_kv_cache_ptr(key: int, size: int) -> int:
    libc = ctypes.CDLL("/usr/lib/x86_64-linux-gnu/libc.so.6", use_errno=True)
    libc.shmget.argtypes = (ctypes.c_long, ctypes.c_size_t, ctypes.c_int)
    libc.shmget.restype = ctypes.c_int
    libc.shmat.argtypes = (ctypes.c_int, ctypes.c_void_p, ctypes.c_int)
    libc.shmat.restype = ctypes.c_void_p

    requested_size = size
    use_hugetlb = enable_huge_page()

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

    interleave_pages_across_numa_nodes(libc, shm_addr, size_to_alloc)

    # Best-effort memory prefaulting in background to speed up subsequent cudaHostRegister
    def _pre_warm_memory():
        page_size = _get_default_hugepage_size() if use_hugetlb else 4096
        arr = np.ctypeslib.as_array(ctypes.cast(shm_addr, ctypes.POINTER(ctypes.c_uint8)), shape=(size_to_alloc,))
        worker_num = 8
        chunk_size = triton.cdiv(size_to_alloc, worker_num * page_size) * page_size

        def _warm_range(worker_id: int):
            start = worker_id * chunk_size
            end = min(size_to_alloc, start + chunk_size)
            return int(arr[start:end:page_size].sum())

        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_num) as executor:
            volatile_sum = sum(executor.map(_warm_range, range(worker_num)))
        logger.info(f"pre warmed shared memory pages successfully, checksum={volatile_sum})")

    th = threading.Thread(target=_pre_warm_memory, name=f"cpu_cache_pre_warm_{key}", daemon=True)
    th.start()

    return shm_addr


@lru_cache(maxsize=None)
def register_shm_ptr_to_pin(shm_ptr: int, size: int) -> int:
    """Synchronously cudaHostRegister the given [shm_ptr, shm_ptr+size)."""
    chunk_bytes = 128 * 1024 * 1024  # 128M性能最好
    tasks: list[tuple[int, int]] = []
    offset = 0
    while offset < size:
        seg_len = min(chunk_bytes, size - offset)
        tasks.append((offset, seg_len))
        offset += seg_len

    cuda = ctypes.CDLL("/usr/local/cuda/targets/x86_64-linux/lib/libcudart.so")
    cuda.cudaHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
    cuda.cudaHostRegister.restype = ctypes.c_int
    cuda.cudaHostGetDevicePointer.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_int]
    cuda.cudaHostGetDevicePointer.restype = ctypes.c_int

    cudaHostRegisterFlag = 3

    device_id = get_current_device_id()
    torch.cuda.set_device(device_id)
    desc = f"pid {os.getpid()} Registering pinned host memory"

    def _register_one_segment(task: Tuple[int, int]):
        offset, seg_len = task
        torch.cuda.set_device(device_id)
        ptr = ctypes.c_void_p(shm_ptr + offset)
        r = cuda.cudaHostRegister(ptr, ctypes.c_size_t(seg_len), cudaHostRegisterFlag)
        if r != 0:
            raise Exception(f"cudaHostRegister failed with error code {r}, prefer to use hugetlb")
        return

    # worker_num的数值需要与_pre_warm_memory一致，不然会丢失warmup的效果
    if tasks:
        worker_num = min(8, len(tasks))
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_num) as executor:
            futures = [executor.submit(_register_one_segment, task) for task in tasks]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=desc):
                future.result()

    device_ptr = ctypes.c_void_p()
    host_ptr = ctypes.c_void_p(shm_ptr)
    res = cuda.cudaHostGetDevicePointer(ctypes.byref(device_ptr), host_ptr, 0)
    if res != 0:
        raise Exception(f"cudaHostGetDevicePointer failed with error code {res}")
    logger.info(f"cudaHostGetDevicePointer success, host_ptr={host_ptr.value}, device_ptr={device_ptr.value}")
    return device_ptr.value


@lru_cache(maxsize=None)
def attach_shm_kv_cache_ptr(key: int, size: int) -> int:
    libc = ctypes.CDLL("/usr/lib/x86_64-linux-gnu/libc.so.6", use_errno=True)
    libc.shmget.argtypes = (ctypes.c_long, ctypes.c_size_t, ctypes.c_int)
    libc.shmget.restype = ctypes.c_int
    libc.shmat.argtypes = (ctypes.c_int, ctypes.c_void_p, ctypes.c_int)
    libc.shmat.restype = ctypes.c_void_p

    # Try to locate an existing SHM without creating a new one
    shmid = libc.shmget(key, 0, 0)
    if shmid < 0:
        shmid = libc.shmget(key, size, 0)
    if shmid < 0:
        err = ctypes.get_errno()
        raise Exception(f"Error locating existing shared memory (errno={err})")

    shm_addr = libc.shmat(shmid, ctypes.c_void_p(0), 0)
    if shm_addr == ctypes.c_void_p(-1).value:
        err = ctypes.get_errno()
        raise Exception(f"Error attaching shared memory (errno={err})")

    logger.info(f"Attached to SHM key={key}, shmid={shmid}, addr={shm_addr}")

    interleave_pages_across_numa_nodes(libc, shm_addr, size)
    return shm_addr


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
    return 2 * 1024 * 1024


def _get_online_numa_nodes() -> List[int]:
    for path in ("/sys/devices/system/node/has_memory", "/sys/devices/system/node/online"):
        try:
            with open(path, "r") as f:
                online = f.read().strip()
            nodes: List[int] = []
            for part in online.split(","):
                if "-" in part:
                    start, end = part.split("-")
                    nodes.extend(range(int(start), int(end) + 1))
                else:
                    nodes.append(int(part))
            return nodes
        except Exception:
            continue
    return [0]


def interleave_pages_across_numa_nodes(libc, addr: int, size: int) -> bool:
    """为 CPU KV cache 的共享内存映射设置 NUMA 交错分配策略。

    CPU KV cache 使用 SysV SHM 在多个进程间共享。默认的 first-touch 策略会把物理页分配到
    首次触页线程所在的 NUMA 节点；在多 Socket 机器上，后台 prefault 线程的调度位置可能导致
    大量 cache 页集中到单个内存控制器，限制多个 GPU 并发 load/offload 的主机内存带宽。

    本函数通过 ``mbind(MPOL_INTERLEAVE)`` 将映射范围内尚未分配的物理页按页偏移交错放置到
    可用 NUMA 节点。调用方应在首次触页前设置策略：creator 在启动 prefault 线程前调用；
    HugeTLB 的共享策略不会可靠地传播到其他进程的 VMA，因此 attacher 也需要在访问映射前调用。

    调用未设置 ``MPOL_MF_MOVE``，所以只影响后续缺页分配，不迁移已经分配的物理页。该功能默认
    关闭，只有设置 ``LIGHTLLM_ENABLE_NUMA_INTERLEAVE`` 后才会启用；未启用、单 NUMA、不支持的
    架构或 syscall 失败都会安全回退到原有 first-touch 行为。

    Args:
        libc: 使用 ``use_errno=True`` 加载的 libc 对象，用于发起 raw ``mbind`` syscall。
        addr: ``shmat`` 返回的、按页对齐的映射起始虚拟地址。
        size: 需要设置策略的映射长度；HugeTLB 模式下会向上对齐到默认大页大小。

    Returns:
        策略成功安装时返回 ``True``；跳过或安装失败时返回 ``False``。
    """
    MPOL_INTERLEAVE = 3
    SYS_MBIND = {"x86_64": 237, "aarch64": 235}.get(os.uname().machine)

    if not enable_cpu_cache_numa_interleave():
        return False

    if SYS_MBIND is None:
        logger.warning(f"unsupported architecture {os.uname().machine}, skip cpu cache numa interleave")
        return False

    if enable_huge_page():
        huge_sz = _get_default_hugepage_size()
        size = triton.cdiv(size, huge_sz) * huge_sz

    def _mbind(mode, mask):
        nodemask = ctypes.c_ulong(mask)
        libc.syscall.restype = ctypes.c_long
        return libc.syscall(
            ctypes.c_long(SYS_MBIND),
            ctypes.c_void_p(addr),
            ctypes.c_ulong(size),
            ctypes.c_int(mode),
            ctypes.byref(nodemask),
            # Raw syscall ABI decrements maxnode before copying the bitmap.
            # Passing mask width + 1 preserves every bit while copying exactly one c_ulong.
            ctypes.c_ulong(ctypes.sizeof(nodemask) * 8 + 1),
            ctypes.c_uint(0),
        )

    nodes = _get_online_numa_nodes()
    if len(nodes) <= 1:
        return False
    if max(nodes) >= 64:
        logger.warning(f"more than 64 numa nodes ({nodes}), skip cpu cache numa interleave")
        return False
    try:
        ret = _mbind(MPOL_INTERLEAVE, sum(1 << n for n in nodes))
        if ret != 0:
            logger.warning(
                f"mbind MPOL_INTERLEAVE failed (errno={ctypes.get_errno()}), "
                f"cpu kv cache pages will use default first-touch numa policy"
            )
            return False
        logger.info(f"cpu kv cache pages interleaved across numa nodes {nodes}")
        return True
    except Exception as e:
        logger.warning(f"cpu cache numa interleave skipped: {e}")
        return False
