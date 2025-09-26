import os
import ctypes
import atexit
import signal
import psutil
from multiprocessing import shared_memory
from typing import Set, Optional
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class AutoShmCleanup:
    def __init__(self):
        self.libc = None
        self._init_libc()
        # System V
        self.registered_shm_keys = []
        self.registered_shm_ids = []
        # POSIX
        self.registered_posix_shm_names = []
        self._register_handlers_for_cleanup()

    def _init_libc(self):
        try:
            self.libc = ctypes.CDLL("/usr/lib/x86_64-linux-gnu/libc.so.6")
            self.libc.shmget.argtypes = (ctypes.c_long, ctypes.c_size_t, ctypes.c_int)
            self.libc.shmget.restype = ctypes.c_int
            self.libc.shmctl.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_void_p)
            self.libc.shmctl.restype = ctypes.c_int
        except Exception as e:
            logger.debug(f"libc init failed: {e}")
            self.libc = None

    def _register_handlers_for_cleanup(self):
        atexit.register(self._cleanup)
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(sig, self._signal_cleanup_handler)

    def _signal_cleanup_handler(self, signum, frame):
        self._cleanup()
        parent = psutil.Process(os.getpid())
        # 递归拿到所有子进程并终止
        for ch in parent.children(recursive=True):
            ch.kill()

    def _cleanup(self):
        """清理：System V 执行 IPC_RMID，POSIX 执行 unlink。"""
        removed_sysv = 0
        IPC_RMID = 0
        for shmid in self.registered_shm_ids:
            try:
                if self.libc.shmctl(shmid, IPC_RMID, None) == 0:
                    removed_sysv += 1
            except Exception as e:
                logger.warning(f"cleanup: shmid {shmid} clean failed, reason: {e}")
                pass
        for key in self.registered_shm_keys:
            shmid = self.libc.shmget(key, 0, 0)
            try:
                if shmid >= 0 and self.libc.shmctl(shmid, IPC_RMID, None) == 0:
                    removed_sysv += 1
            except Exception as e:
                logger.warning(f"cleanup: shmid {shmid} clean failed, reason: {e}")
                pass
        if removed_sysv:
            logger.info(f"cleanup: removed {removed_sysv} System V shm segments")

        removed_posix = 0
        for name in self.registered_posix_shm_names:
            try:
                shm = shared_memory.SharedMemory(name=name, create=False)
                try:
                    shm.unlink()
                    removed_posix += 1
                except FileNotFoundError:
                    pass
                except Exception as e:
                    logger.warning(f"cleanup: posix shm {name} clean failed, reason: {e}")
                    pass
                finally:
                    shm.close()
            except FileNotFoundError:
                pass
            except Exception as e:
                logger.warning(f"cleanup: posix {name} clean failed, reason: {e}")
                pass
        if removed_posix:
            logger.info(f"cleanup: unlinked {removed_posix} POSIX shm segments")

    def register_sysv_shm(self, key: int, shmid: Optional[int] = None):
        """注册 System V 共享内存。"""
        self.registered_shm_keys.append(key)
        if shmid is not None:
            self.registered_shm_ids.append(shmid)
        return

    def register_posix_shm(self, name: str):
        """注册 POSIX 共享内存。"""
        self.registered_posix_shm_names.append(name)
        return


# 全局自动清理器实例
_auto_cleanup = None
def get_auto_cleanup() -> AutoShmCleanup:
    """获取全局自动清理器实例"""
    global _auto_cleanup
    if _auto_cleanup is None:
        _auto_cleanup = AutoShmCleanup()
    return _auto_cleanup


def register_sysv_shm_for_cleanup(key: int, shmid: Optional[int] = None):
    get_auto_cleanup().register_sysv_shm(key, shmid)


def register_posix_shm_for_cleanup(name: str):
    get_auto_cleanup().register_posix_shm(name)
