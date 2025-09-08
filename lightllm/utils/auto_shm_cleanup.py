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
        # 使用atexit确保正常退出时清理
        atexit.register(self._cleanup_all)

        # 注册信号处理器处理异常退出
        signal.signal(signal.SIGTERM, self._signal_cleanup_handler)
        signal.signal(signal.SIGINT, self._signal_cleanup_handler)
        signal.signal(signal.SIGHUP, self._signal_cleanup_handler)

        logger.info("Auto shared memory cleanup handlers registered")

    def _signal_cleanup_handler(self, signum, frame):
        """信号处理函数 - 异常退出时清理"""
        logger.info(f"Process received signal {signum}, cleaning up shared memory...")
        self._cleanup_all()
        # 继续原有的信号处理流程
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    def register_sysv_shm(self, key: int, shmid: Optional[int] = None):
        """注册System V共享内存用于自动清理"""
        self.registered_shm_keys.add(key)
        if shmid is not None:
            self.registered_shm_ids.add(shmid)
        logger.debug(f"Registered System V shm key {key} for auto cleanup")

    def register_posix_shm(self, name: str):
        """注册POSIX共享内存用于自动清理"""
        self.registered_posix_shm_names.add(name)
        logger.debug(f"Registered POSIX shm {name} for auto cleanup")

    def _cleanup_all(self):
        """清理所有注册的共享内存"""
        cleaned_count = 0

        # 清理System V共享内存
        if self.libc:
            # 通过shmid清理
            for shmid in list(self.registered_shm_ids):
                try:
                    result = self.libc.shmctl(shmid, 0, None)  # IPC_RMID = 0
                    if result == 0:
                        logger.info(f"Auto cleanup: removed System V shm segment {shmid}")
                        cleaned_count += 1
                except Exception as e:
                    logger.debug(f"Failed to cleanup System V shm segment {shmid}: {e}")

            # 通过key清理
            for key in list(self.registered_shm_keys):
                try:
                    shmid = self.libc.shmget(key, 0, 0)
                    if shmid >= 0:
                        result = self.libc.shmctl(shmid, 0, None)  # IPC_RMID = 0
                        if result == 0:
                            logger.info(f"Auto cleanup: removed System V shm key {key} (shmid {shmid})")
                            cleaned_count += 1
                except Exception as e:
                    logger.debug(f"Failed to cleanup System V shm key {key}: {e}")

        # 清理POSIX共享内存
        for name in list(self.registered_posix_shm_names):
            try:
                shm = shared_memory.SharedMemory(name=name, create=False)
                shm.close()
                shm.unlink()
                logger.info(f"Auto cleanup: removed POSIX shm {name}")
                cleaned_count += 1
            except FileNotFoundError:
                logger.debug(f"POSIX shm {name} already removed")
            except Exception as e:
                logger.debug(f"Failed to cleanup POSIX shm {name}: {e}")

        if cleaned_count > 0:
            logger.info(f"Auto cleanup completed: removed {cleaned_count} shared memory segments")

        # 清空注册列表
        self.registered_shm_keys.clear()
        self.registered_shm_ids.clear()
        self.registered_posix_shm_names.clear()


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
