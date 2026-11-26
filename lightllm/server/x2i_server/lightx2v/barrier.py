
import ctypes
import ctypes.util
import platform
from lightllm.utils.shm_utils import create_or_link_shm


# ============================================================
# Linux futex + libatomic
# ============================================================

FUTEX_WAIT = 0
FUTEX_WAKE = 1

# GCC atomic memory order
MEMORY_ORDER_ACQUIRE = 2
MEMORY_ORDER_RELEASE = 3


def get_futex_syscall_number():
    machine = platform.machine().lower()

    if machine in ("x86_64", "amd64"):
        return 202
    if machine in ("aarch64", "arm64"):
        return 98

    raise RuntimeError(f"Unsupported architecture: {machine}")


LIBC = ctypes.CDLL("libc.so.6", use_errno=True)
LIBC.syscall.restype = ctypes.c_long

LIBATOMIC_PATH = ctypes.util.find_library("atomic")
if not LIBATOMIC_PATH:
    raise RuntimeError("libatomic.so.1 not found")

LIBATOMIC = ctypes.CDLL(LIBATOMIC_PATH)

atomic_load_4 = LIBATOMIC.__atomic_load_4
atomic_load_4.argtypes = [ctypes.c_void_p, ctypes.c_int]
atomic_load_4.restype = ctypes.c_uint32

atomic_store_4 = LIBATOMIC.__atomic_store_4
atomic_store_4.argtypes = [
    ctypes.c_void_p,
    ctypes.c_uint32,
    ctypes.c_int,
]
atomic_store_4.restype = None

SYS_FUTEX = get_futex_syscall_number()


def atomic_load_acquire(cell):
    return int(
        atomic_load_4(
            ctypes.cast(ctypes.byref(cell), ctypes.c_void_p),
            MEMORY_ORDER_ACQUIRE,
        )
    )


def atomic_store_release(cell, value):
    atomic_store_4(
        ctypes.cast(ctypes.byref(cell), ctypes.c_void_p),
        ctypes.c_uint32(value),
        MEMORY_ORDER_RELEASE,
    )


def futex_wait(cell, expected):
    """
    等待 cell 的值发生变化。
    使用普通 FUTEX_WAIT，不带 PRIVATE_FLAG，
    因为 P2/P3 是不同进程。
    """
    ret = LIBC.syscall(
        SYS_FUTEX,
        ctypes.byref(cell),
        FUTEX_WAIT,
        ctypes.c_uint32(expected),
        0,
        0,
        0,
    )

    if ret == -1:
        err = ctypes.get_errno()

        # EAGAIN: value 已经变化
        # EINTR: 被信号中断
        if err not in (4, 11):
            raise OSError(err, "futex_wait failed")


def futex_wake(cell, count=1):
    ret = LIBC.syscall(
        SYS_FUTEX,
        ctypes.byref(cell),
        FUTEX_WAKE,
        ctypes.c_int(count),
        0,
        0,
        0,
    )

    if ret == -1:
        err = ctypes.get_errno()
        raise OSError(err, "futex_wake failed")


def wait_until_at_least(cell, target):
    while True:
        value = atomic_load_acquire(cell)

        if value >= target:
            return

        # expected 使用刚刚读到的值。
        # 如果值在 load 和 wait 之间发生变化，
        # FUTEX_WAIT 会返回 EAGAIN，不会丢失唤醒。
        futex_wait(cell, value)


# ============================================================
# Shared memory layout
# ============================================================
class SharedWords:
    def __init__(self, name):
        self.shm = create_or_link_shm(name, 8)
        self.snapshot = ctypes.c_int32.from_buffer(self.shm.buf, 0)
        self.snapshot_seq = ctypes.c_uint32.from_buffer(self.shm.buf, 4)
        self.reset()

    def reset(self):
        self.snapshot.value = 0
        self.snapshot_seq.value = 0

    def release(self):
        del self.snapshot
        del self.snapshot_seq
