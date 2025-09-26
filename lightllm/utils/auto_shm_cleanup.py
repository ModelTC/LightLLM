#!/usr/bin/env python3
"""
共享内存自动清理工具

LightLLM进程退出时自动清理共享内存，防止内存泄漏。
"""

import os
import sys
import ctypes
import atexit
import signal
from multiprocessing import shared_memory
from typing import Set, Optional
from lightllm.utils.log_utils import init_logger
from lightllm.utils.envs_utils import get_env_start_args

logger = init_logger(__name__)


class AutoShmCleanup:
    """自动共享内存清理器 - 进程退出时自动清理"""

    def __init__(self):
        self.libc = None
        self._init_libc()
        # System V共享内存
        self.registered_shm_keys: Set[int] = set()
        self.registered_shm_ids: Set[int] = set()
        # 附加地址（用于在清理时执行 shmdt）
        self.registered_shm_addrs: Set[int] = set()
        # POSIX共享内存
        self.registered_posix_shm_names: Set[str] = set()
        self._auto_register_handlers()

    def _init_libc(self):
        """初始化libc库"""
        try:
            self.libc = ctypes.CDLL("/usr/lib/x86_64-linux-gnu/libc.so.6")

            # 设置函数签名
            self.libc.shmget.argtypes = (ctypes.c_long, ctypes.c_size_t, ctypes.c_int)
            self.libc.shmget.restype = ctypes.c_int

            self.libc.shmctl.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_void_p)
            self.libc.shmctl.restype = ctypes.c_int

            self.libc.shmdt.argtypes = (ctypes.c_void_p,)
            self.libc.shmdt.restype = ctypes.c_int

        except Exception as e:
            logger.error(f"Failed to initialize libc: {e}")
            self.libc = None

    def _auto_register_handlers(self):
        """自动注册清理处理程序"""
        # atexit：始终使用快速清理，避免长时间阻塞
        atexit.register(self._cleanup_fast)

        # 注册信号处理器处理异常退出
        signal.signal(signal.SIGTERM, self._signal_cleanup_handler)
        signal.signal(signal.SIGINT, self._signal_cleanup_handler)
        signal.signal(signal.SIGHUP, self._signal_cleanup_handler)

        # 注册 SIGCHLD 处理器，及时回收子进程，避免产生僵尸
        def _sigchld_handler(signum, frame):
            self._reap_children_nonblock()

        try:
            signal.signal(signal.SIGCHLD, _sigchld_handler)
        except Exception as e:
            logger.debug(f"SIGCHLD handler not installed: {e}")

        logger.info("Auto shared memory cleanup handlers registered")

    def _reap_children_nonblock(self):
        """非阻塞地回收所有已退出的子进程，避免僵尸堆积。"""
        try:
            while True:
                try:
                    pid, _ = os.waitpid(-1, os.WNOHANG)
                except ChildProcessError:
                    break
                if pid == 0:
                    break
                else:
                    logger.debug(f"Reaped child process pid={pid}")
        except Exception as e:
            logger.debug(f"Non-block reap failed: {e}")

    def _signal_cleanup_handler(self, signum, frame):
        """信号处理函数 - 异常退出时清理"""
        logger.info(f"Process received signal {signum}, cleaning up shared memory (fast)...")
        try:
            self._cleanup_fast()
        except Exception as e:
            logger.debug(f"signal cleanup error: {e}")
        # 在退出前，尽可能终止并回收子进程，避免留下僵尸
        try:
            try:
                import psutil  # 该项目其他模块已使用，通常可用

                parent = psutil.Process(os.getpid())
                children = parent.children(recursive=True)
                for ch in children:
                    try:
                        ch.kill()
                    except Exception:
                        pass
            except Exception as e:
                logger.debug(f"psutil children kill skipped: {e}")
            # 非阻塞回收一次
            self._reap_children_nonblock()
        except Exception as e:
            logger.debug(f"pre-exit reap skipped: {e}")
        # 立即退出，绕过 Python/CUDA 析构，确保秒退
        os._exit(128 + int(signum))

    def _cleanup_fast(self):
        """快速清理：
        - 不执行 shmdt（避免对超大映射分离的耗时）
        - 仅对已知 shmid 执行 shmctl(IPC_RMID) 标记删除
        - 不做通过 key 的额外扫描
        """
        if not self.libc:
            return
        removed = 0
        for shmid in list(self.registered_shm_ids):
            try:
                result = self.libc.shmctl(shmid, 0, None)  # IPC_RMID = 0
                if result == 0:
                    removed += 1
            except Exception:
                pass
        if removed:
            logger.info(f"Fast cleanup: IPC_RMID set for {removed} System V shm segments")

    def register_sysv_shm(self, key: int, shmid: Optional[int] = None):
        """注册System V共享内存用于自动清理"""
        self.registered_shm_keys.add(key)
        if shmid is not None:
            self.registered_shm_ids.add(shmid)
        logger.debug(f"Registered System V shm key {key} for auto cleanup")

    def register_sysv_shm_addr(self, addr: int):
        """记录已附加的System V共享内存地址（当前快速清理不执行shmdt，仅用于将来扩展或调试）"""
        try:
            if addr is not None and int(addr) != 0 and int(addr) != -1:
                self.registered_shm_addrs.add(int(addr))
                logger.debug(f"Registered System V shm addr {addr} for auto cleanup")
        except Exception:
            pass

    def register_posix_shm(self, name: str):
        """注册POSIX共享内存用于自动清理"""
        self.registered_posix_shm_names.add(name)
        logger.debug(f"Registered POSIX shm {name} for auto cleanup")



def auto_register_sysv_shm_addr(addr: int):
    """对外提供注册已附加地址的便捷接口"""
    cleanup = get_auto_cleanup()
    cleanup.register_sysv_shm_addr(addr)


# 全局自动清理器实例
_auto_cleanup = None


def get_auto_cleanup() -> AutoShmCleanup:
    """获取全局自动清理器实例"""
    global _auto_cleanup
    if _auto_cleanup is None:
        _auto_cleanup = AutoShmCleanup()
    return _auto_cleanup


def auto_register_cpu_cache():
    """自动注册CPU Cache共享内存清理"""
    try:
        args = get_env_start_args()
        if hasattr(args, "cpu_kv_cache_shm_id"):
            cleanup = get_auto_cleanup()
            cleanup.register_sysv_shm(args.cpu_kv_cache_shm_id)
            logger.info(f"Auto registered CPU cache shm key {args.cpu_kv_cache_shm_id}")
    except Exception as e:
        logger.error(f"Failed to auto register CPU cache shm: {e}")


def auto_register_sysv_shm(key: int, shmid: Optional[int] = None):
    """自动注册System V共享内存清理"""
    cleanup = get_auto_cleanup()
    cleanup.register_sysv_shm(key, shmid)


def auto_register_posix_shm(name: str):
    """自动注册POSIX共享内存清理"""
    cleanup = get_auto_cleanup()
    cleanup.register_posix_shm(name)


# 增强的共享内存创建函数，自动注册清理
def create_auto_cleanup_sysv_shm(key: int, size: int) -> int:
    """创建System V共享内存并自动注册清理"""
    try:
        libc = ctypes.CDLL("/usr/lib/x86_64-linux-gnu/libc.so.6")
        libc.shmget.argtypes = (ctypes.c_long, ctypes.c_size_t, ctypes.c_int)
        libc.shmget.restype = ctypes.c_int
        libc.shmat.argtypes = (ctypes.c_int, ctypes.c_void_p, ctypes.c_int)
        libc.shmat.restype = ctypes.c_void_p

        # 创建共享内存
        shmflg = 0o666 | 0o1000  # 权限和 IPC_CREAT 标志
        shmid = libc.shmget(key, size, shmflg)

        if shmid < 0:
            raise Exception(f"Failed to create System V shared memory with key {key}")

        # 附加共享内存
        shm_addr = libc.shmat(shmid, ctypes.c_void_p(0), 0)

        if shm_addr == ctypes.c_void_p(-1).value:
            raise Exception(f"Failed to attach System V shared memory {shmid}")

        # 自动注册清理
        auto_register_sysv_shm(key, shmid)
        logger.info(f"Created and auto-registered System V shm key {key}, shmid {shmid}")

        return shm_addr

    except Exception as e:
        logger.error(f"Failed to create auto-cleanup System V shm: {e}")
        raise


def create_auto_cleanup_posix_shm(name: str, size: int, create: bool = True) -> shared_memory.SharedMemory:
    """创建POSIX共享内存并自动注册清理"""
    try:
        if create:
            try:
                shm = shared_memory.SharedMemory(name=name, create=True, size=size)
            except FileExistsError:
                # 如果已存在，尝试连接
                shm = shared_memory.SharedMemory(name=name, create=False)
        else:
            shm = shared_memory.SharedMemory(name=name, create=False)

        # 自动注册清理
        auto_register_posix_shm(name)
        logger.info(f"Created and auto-registered POSIX shm {name}")

        return shm

    except Exception as e:
        logger.error(f"Failed to create auto-cleanup POSIX shm {name}: {e}")
        raise


# 向后兼容的别名（保留原有的接口）
register_shm_cleanup = auto_register_sysv_shm
register_posix_shm_cleanup = auto_register_posix_shm
register_cpu_cache_shm_cleanup = auto_register_cpu_cache
