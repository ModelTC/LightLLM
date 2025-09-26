import ctypes
import dataclasses
import os
import xxhash
import threading
import time
import numpy as np
from functools import lru_cache
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.log_utils import init_logger
from lightllm.utils.config_utils import get_config_json
from typing import List, Tuple, Optional
from tqdm import tqdm

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


class AsyncRegistrationHandle:
    """A handle for async host memory registration.

    - wait(): blocks until registration finishes, prints tqdm progress, and returns device pointer (int).
    - is_done(): non-blocking completion check.
    - result: device pointer if done, otherwise None.
    """

    def __init__(self, total_chunks: int, desc: str):
        self._total = total_chunks
        self._desc = desc
        self._done = 0
        self._done_lock = threading.Lock()
        self._exception: Optional[BaseException] = None
        self._result: Optional[int] = None
        self._thread: Optional[threading.Thread] = None
        self._done_event = threading.Event()

    def _inc_done(self, n: int = 1):
        with self._done_lock:
            self._done += n

    def _set_exception(self, exc: BaseException):
        self._exception = exc

    def _set_result(self, ptr: int):
        self._result = ptr

    def set_thread(self, th: threading.Thread):
        self._thread = th

    def is_done(self) -> bool:
        return self._done_event.is_set()

    @property
    def result(self) -> Optional[int]:
        return self._result

    def wait(self) -> int:
        """Block until the async registration completes. Only here we print tqdm progress."""
        last_done = 0
        with tqdm(total=self._total, desc=self._desc) as pbar:
            while not self._done_event.is_set():
                with self._done_lock:
                    cur = self._done
                if cur > last_done:
                    pbar.update(cur - last_done)
                    last_done = cur
                time.sleep(0.1)
            # final update
            with self._done_lock:
                cur = self._done
            if cur > last_done:
                pbar.update(cur - last_done)
                last_done = cur

        if self._thread is not None and self._thread.is_alive():
            self._thread.join()

        if self._exception is not None:
            raise self._exception
        assert self._result is not None, "registration finished without result"
        return self._result


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
    libc = ctypes.CDLL("/usr/lib/x86_64-linux-gnu/libc.so.6", use_errno=True)

    # 设置 shmget 函数的参数类型和返回类型
    libc.shmget.argtypes = (ctypes.c_long, ctypes.c_size_t, ctypes.c_int)
    libc.shmget.restype = ctypes.c_int

    # 设置 shmat 函数的参数类型和返回类型
    libc.shmat.argtypes = (ctypes.c_int, ctypes.c_void_p, ctypes.c_int)
    libc.shmat.restype = ctypes.c_void_p

    # 创建共享内存
    key = args.cpu_kv_cache_shm_id  # 共享内存的键
    requested_size = calcu_cpu_cache_meta().calcu_size()  # 共享内存大小

    # 固定使用大页 SHM_HUGETLB（针对数百 GB 场景的激进配置）
    use_hugetlb = True

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

    # 向上对齐到大页大小
    huge_sz = _get_default_hugepage_size()
    size_to_alloc = ((requested_size + huge_sz - 1) // huge_sz) * huge_sz
    shmflg = 0o666 | 0o1000  # 权限和 IPC_CREAT 标志
    used_hugetlb = False
    if use_hugetlb:
        # SHM_HUGETLB = 0o4000
        SHM_HUGETLB = 0o4000
        shmflg |= SHM_HUGETLB
        logger.info(
            f"Using SHM_HUGETLB, hugepage_size={huge_sz} bytes, requested={requested_size}, alloc={size_to_alloc}"
        )

    # 优先尝试 HugeTLB 分配，失败则回退到普通页
    shmid = libc.shmget(key, size_to_alloc, shmflg)
    alloc_size = size_to_alloc
    if shmid >= 0 and use_hugetlb:
        used_hugetlb = True
    if shmid < 0 and use_hugetlb:
        err = ctypes.get_errno()
        try:
            required_pages = size_to_alloc // huge_sz
            logger.error(
                f"shmget with SHM_HUGETLB failed (errno={err}). Required hugepages (~{required_pages}) "
                f"may exceed available pool. Falling back to regular pages."
            )
        except Exception:
            logger.error(f"shmget with SHM_HUGETLB failed (errno={err}). Falling back to regular pages.")
        # 回退：去掉 HUGETLB 标志，使用请求原始大小
        shmflg_n = 0o666 | 0o1000
        shmid = libc.shmget(key, size_to_alloc, shmflg_n)
        used_hugetlb = False

    if shmid < 0:
        err = ctypes.get_errno()
        raise Exception(f"Error creating shared memory (errno={err})")

    logger.info(f"Shared memory ID: {shmid}")

    # 附加共享内存
    shm_addr = libc.shmat(shmid, ctypes.c_void_p(0), 0)

    if shm_addr == ctypes.c_void_p(-1).value:
        raise Exception("Error attaching shared memory")

    logger.info(f"Shared memory attached at address: {shm_addr}")

    # 非 HUGETLB 情况下，给予内核 WILLNEED/HUGEPAGE 提示与轻量预触碰
    if not used_hugetlb:
        try:
            libc.madvise.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
            libc.madvise.restype = ctypes.c_int
            MADV_WILLNEED = 3
            MADV_HUGEPAGE = 14
            r = libc.madvise(ctypes.c_void_p(shm_addr), ctypes.c_size_t(alloc_size), MADV_WILLNEED)
            if r != 0:
                logger.debug(f"madvise WILLNEED return {r}")
            r = libc.madvise(ctypes.c_void_p(shm_addr), ctypes.c_size_t(alloc_size), MADV_HUGEPAGE)
            if r != 0:
                logger.debug(f"madvise HUGEPAGE return {r}")
        except Exception as e:
            logger.debug(f"madvise not applied: {e}")

        try:
            try:
                libc.mlock2
                libc.mlock2.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
                libc.mlock2.restype = ctypes.c_int
                MLOCK_ONFAULT = 1
                r = libc.mlock2(ctypes.c_void_p(shm_addr), ctypes.c_size_t(alloc_size), MLOCK_ONFAULT)
                if r != 0:
                    logger.debug(f"mlock2(MLOCK_ONFAULT) failed code={r}, skip prefault")
                else:
                    logger.info("mlock2(MLOCK_ONFAULT) set successfully")
            except AttributeError:
                logger.info("mlock2 not available; skip prefault to avoid long blocking on huge regions")
        except Exception as e:
            logger.debug(f"Prefault step skipped: {e}")

    # 自动注册清理
    try:
        from lightllm.utils.auto_shm_cleanup import auto_register_sysv_shm, auto_register_sysv_shm_addr

        auto_register_sysv_shm(key, shmid)
        auto_register_sysv_shm_addr(int(shm_addr))
        logger.info(f"Auto-registered shared memory key {key} (shmid {shmid}) and addr {hex(shm_addr)} for cleanup")
    except Exception as e:
        logger.warning(f"Failed to register auto shm cleanup: {e}")

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


@lru_cache(maxsize=None)
def register_shm_ptr_to_pin(shm_ptr: int, size: int) -> AsyncRegistrationHandle:
    """Start async cudaHostRegister on the given [shm_ptr, shm_ptr+size) and return a handle.

    Registration uses single-threaded, ascending chunks to preserve device VA locality.
    Progress bar is printed only when handle.wait() is called.
    """
    # compute chunks
    try:
        page_sz = os.sysconf("SC_PAGE_SIZE")
    except Exception:
        page_sz = 4096

    chunk_bytes = max(page_sz, 128 * 1024 * 1024)  # 1GiB
    chunk_bytes = (chunk_bytes // page_sz) * page_sz
    if chunk_bytes <= 0:
        chunk_bytes = page_sz

    tasks: list[tuple[int, int]] = []
    off = 0
    while off < size:
        seg = min(chunk_bytes, size - off)
        tasks.append((off, seg))
        off += seg

    desc = f"pid {os.getpid()} Registering pinned host memory (async)"
    handle = AsyncRegistrationHandle(total_chunks=len(tasks), desc=desc)

    def _worker():
        try:
            cuda = ctypes.CDLL("/usr/local/cuda/targets/x86_64-linux/lib/libcudart.so")
            cuda.cudaHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
            cuda.cudaHostRegister.restype = ctypes.c_int
            cuda.cudaHostGetDevicePointer.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_int]
            cuda.cudaHostGetDevicePointer.restype = ctypes.c_int

            cudaHostRegisterDefault = 0

            try:
                import torch
                from lightllm.utils.dist_utils import get_current_device_id

                torch.cuda.set_device(get_current_device_id())
            except Exception as e:
                logger.debug(f"async set_device skipped: {e}")
            try:
                cuda.cudaFree.argtypes = [ctypes.c_void_p]
                cuda.cudaFree.restype = ctypes.c_int
                _ = cuda.cudaFree(ctypes.c_void_p(0))
            except Exception as e:
                logger.debug(f"async cudaFree(0) skipped: {e}")

            errors = []
            for off, seg in tasks:
                ptr = ctypes.c_void_p(shm_ptr + off)
                r = cuda.cudaHostRegister(ptr, ctypes.c_size_t(seg), cudaHostRegisterDefault)
                if r != 0:
                    logger.error(f"cudaHostRegister failed: offset={off}, size={seg}, err={r}")
                    errors.append((off, r))
                handle._inc_done(1)

            if errors:
                first_err = errors[0][1]
                handle._set_exception(Exception(f"Error registering host memory (multi-chunk). First error code: {first_err}"))
                return

            device_ptr = ctypes.c_void_p()
            host_ptr = ctypes.c_void_p(shm_ptr)
            res = cuda.cudaHostGetDevicePointer(ctypes.byref(device_ptr), host_ptr, 0)
            if res != 0:
                handle._set_exception(RuntimeError(f"cudaHostGetDevicePointer failed with error code {res}"))
                return

            handle._set_result(int(device_ptr.value))
        except BaseException as e:
            handle._set_exception(e)
        finally:
            handle._done_event.set()

    th = threading.Thread(target=_worker, name="kv-cache-register", daemon=True)
    handle.set_thread(th)
    th.start()
    return handle


def register_shm_ptr_to_pin_sync(shm_ptr: int, size: int) -> int:
    """Synchronous wrapper: start async registration and wait for completion, returning device pointer."""
    return register_shm_ptr_to_pin(shm_ptr, size).wait()


@lru_cache(maxsize=None)
def attach_shm_kv_cache_ptr() -> int:
    """Attach to an existing SysV SHM segment by key and return its address.

    Only attaches; it will not create the segment or perform madvise/mlock.
    """
    args = get_env_start_args()
    key = args.cpu_kv_cache_shm_id

    libc = ctypes.CDLL("/usr/lib/x86_64-linux-gnu/libc.so.6", use_errno=True)
    libc.shmget.argtypes = (ctypes.c_long, ctypes.c_size_t, ctypes.c_int)
    libc.shmget.restype = ctypes.c_int
    libc.shmat.argtypes = (ctypes.c_int, ctypes.c_void_p, ctypes.c_int)
    libc.shmat.restype = ctypes.c_void_p

    # Try to locate an existing SHM without creating a new one
    shmid = libc.shmget(key, 0, 0)
    if shmid < 0:
        # Some systems require size > 0 even when not creating; retry with expected size
        try:
            req_size = calcu_cpu_cache_meta().calcu_size()
        except Exception:
            req_size = 1
        shmid = libc.shmget(key, req_size, 0)
    if shmid < 0:
        err = ctypes.get_errno()
        raise Exception(f"Error locating existing shared memory (errno={err})")

    shm_addr = libc.shmat(shmid, ctypes.c_void_p(0), 0)
    if shm_addr == ctypes.c_void_p(-1).value:
        err = ctypes.get_errno()
        raise Exception(f"Error attaching shared memory (errno={err})")

    logger.info(f"Attached to SHM key={key}, shmid={shmid}, addr={shm_addr}")
    # 注册附加地址用于退出时shmdt
    try:
        from lightllm.utils.auto_shm_cleanup import auto_register_sysv_shm_addr, auto_register_sysv_shm
        auto_register_sysv_shm(key, shmid)
        auto_register_sysv_shm_addr(int(shm_addr))
    except Exception as e:
        logger.debug(f"auto_register_sysv_shm_addr failed: {e}")
    return shm_addr
