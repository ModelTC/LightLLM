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
import weakref
from multiprocessing import shared_memory
from typing import Set, Optional
from lightllm.utils.log_utils import init_logger
from lightllm.utils.envs_utils import get_unique_server_name, get_env_start_args

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
        if hasattr(args, 'cpu_kv_cache_shm_id'):
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

    def register_shm_key(self, key: int):
        """注册要清理的System V共享内存键"""
        self.registered_shm_keys.add(key)
        try:
            # 尝试获取对应的shmid
            if self.libc:
                shmid = self.libc.shmget(key, 0, 0)
                if shmid >= 0:
                    self.registered_shm_ids.add(shmid)
                    logger.info(f"Registered shm key {key} with shmid {shmid} for cleanup")
        except Exception as e:
            logger.warning(f"Failed to get shmid for key {key}: {e}")

    def register_posix_shm_name(self, name: str):
        """注册要清理的POSIX共享内存名称"""
        self.registered_posix_shm_names.add(name)
        logger.info(f"Registered POSIX shm name {name} for cleanup")

    def register_cleanup_handlers(self):
        """注册清理处理程序"""
        if self.cleanup_registered:
            return
            
        # 注册atexit清理
        atexit.register(self.cleanup_on_exit)
        
        # 注册信号处理
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGUSR1, self._signal_handler)  # 自定义清理信号
        
        self.cleanup_registered = True
        logger.info("Shared memory cleanup handlers registered")

    def _signal_handler(self, signum, frame):
        """信号处理函数"""
        logger.info(f"Received signal {signum}, cleaning up shared memory...")
        self.cleanup_shared_memory()
        
        # 对于SIGTERM和SIGINT，执行完清理后正常退出
        if signum in (signal.SIGTERM, signal.SIGINT):
            sys.exit(0)

    def cleanup_on_exit(self):
        """退出时清理函数"""
        logger.info("Process exiting, cleaning up shared memory...")
        self.cleanup_shared_memory()

    def cleanup_shared_memory(self):
        """清理注册的共享内存"""
        cleaned_count = 0
        
        # 清理System V共享内存
        cleaned_count += self._cleanup_sysv_shm()
        
        # 清理POSIX共享内存
        cleaned_count += self._cleanup_posix_shm()
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} shared memory segments")
        
        # 清空注册列表
        self.registered_shm_keys.clear()
        self.registered_shm_ids.clear()
        self.registered_posix_shm_names.clear()

    def _cleanup_sysv_shm(self) -> int:
        """清理System V共享内存"""
        if not self.libc:
            logger.error("libc not available for System V shm cleanup")
            return 0
            
        cleaned_count = 0
        
        # 清理通过shmid注册的共享内存
        for shmid in list(self.registered_shm_ids):
            try:
                result = self.libc.shmctl(shmid, 0, None)  # IPC_RMID = 0
                if result == 0:
                    logger.info(f"Successfully removed System V shared memory segment {shmid}")
                    cleaned_count += 1
                else:
                    logger.warning(f"Failed to remove System V shared memory segment {shmid}")
            except Exception as e:
                logger.error(f"Error removing System V shared memory segment {shmid}: {e}")
        
        # 清理通过key注册的共享内存
        for key in list(self.registered_shm_keys):
            try:
                shmid = self.libc.shmget(key, 0, 0)
                if shmid >= 0:
                    result = self.libc.shmctl(shmid, 0, None)  # IPC_RMID = 0
                    if result == 0:
                        logger.info(f"Successfully removed System V shared memory key {key} (shmid {shmid})")
                        cleaned_count += 1
                    else:
                        logger.warning(f"Failed to remove System V shared memory key {key} (shmid {shmid})")
                else:
                    logger.debug(f"System V shared memory key {key} not found or already removed")
            except Exception as e:
                logger.error(f"Error removing System V shared memory key {key}: {e}")
        
        return cleaned_count

    def _cleanup_posix_shm(self) -> int:
        """清理POSIX共享内存"""
        cleaned_count = 0
        
        for name in list(self.registered_posix_shm_names):
            try:
                # 尝试打开并删除POSIX共享内存
                shm = shared_memory.SharedMemory(name=name, create=False)
                shm.close()
                shm.unlink()
                logger.info(f"Successfully removed POSIX shared memory {name}")
                cleaned_count += 1
            except FileNotFoundError:
                logger.debug(f"POSIX shared memory {name} not found or already removed")
            except Exception as e:
                logger.error(f"Error removing POSIX shared memory {name}: {e}")
        
        return cleaned_count

    def force_cleanup_by_pattern(self, pattern: str = None):
        """强制清理匹配模式的共享内存"""
        if pattern is None:
            pattern = get_unique_server_name()
            
        cleaned_count = 0
        
        # 清理System V共享内存
        cleaned_count += self._force_cleanup_sysv_shm(pattern)
        
        # 清理POSIX共享内存
        cleaned_count += self._force_cleanup_posix_shm(pattern)
        
        logger.info(f"Force cleanup completed, removed {cleaned_count} segments")

    def _force_cleanup_sysv_shm(self, pattern: str) -> int:
        """强制清理System V共享内存"""
        if not self.libc:
            return 0
            
        try:
            # 使用ipcs命令获取所有共享内存信息
            result = subprocess.run(['ipcs', '-m'], capture_output=True, text=True)
            if result.returncode != 0:
                logger.error("Failed to run ipcs command")
                return 0
                
            lines = result.stdout.strip().split('\n')
            cleaned_count = 0
            
            for line in lines[3:]:  # 跳过标题行
                parts = line.split()
                if len(parts) >= 6:
                    try:
                        shmid = int(parts[1])
                        # 检查是否是当前用户的共享内存
                        owner = parts[2] if len(parts) > 2 else ""
                        current_user = os.getenv('USER', '')
                        
                        if owner == current_user or not owner:
                            result = self.libc.shmctl(shmid, 0, None)  # IPC_RMID = 0
                            if result == 0:
                                logger.info(f"Force removed System V shared memory segment {shmid}")
                                cleaned_count += 1
                    except (ValueError, Exception) as e:
                        continue
                        
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Error in force cleanup System V shm: {e}")
            return 0

    def _force_cleanup_posix_shm(self, pattern: str) -> int:
        """强制清理POSIX共享内存"""
        try:
            # POSIX共享内存通常位于/dev/shm/目录下
            shm_dir = "/dev/shm"
            if not os.path.exists(shm_dir):
                return 0
                
            cleaned_count = 0
            for filename in os.listdir(shm_dir):
                if pattern in filename:
                    try:
                        shm = shared_memory.SharedMemory(name=filename, create=False)
                        shm.close()
                        shm.unlink()
                        logger.info(f"Force removed POSIX shared memory {filename}")
                        cleaned_count += 1
                    except Exception as e:
                        logger.debug(f"Failed to remove POSIX shm {filename}: {e}")
                        
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Error in force cleanup POSIX shm: {e}")
            return 0


# 全局清理管理器实例
_cleanup_manager = None


def get_cleanup_manager() -> ShmCleanupManager:
    """获取全局清理管理器实例"""
    global _cleanup_manager
    if _cleanup_manager is None:
        _cleanup_manager = ShmCleanupManager()
    return _cleanup_manager


def register_cpu_cache_shm_cleanup():
    """注册CPU Cache共享内存的清理"""
    try:
        args = get_env_start_args()
        if hasattr(args, 'cpu_kv_cache_shm_id'):
            manager = get_cleanup_manager()
            manager.register_shm_key(args.cpu_kv_cache_shm_id)
            manager.register_cleanup_handlers()
            logger.info(f"Registered CPU cache shm key {args.cpu_kv_cache_shm_id} for cleanup")
    except Exception as e:
        logger.error(f"Failed to register CPU cache shm cleanup: {e}")


def register_shm_cleanup(key: int):
    """注册指定的System V共享内存键用于清理"""
    manager = get_cleanup_manager()
    manager.register_shm_key(key)
    manager.register_cleanup_handlers()


def register_posix_shm_cleanup(name: str):
    """注册指定的POSIX共享内存名称用于清理"""
    manager = get_cleanup_manager()
    manager.register_posix_shm_name(name)
    manager.register_cleanup_handlers()


def manual_cleanup():
    """手动触发清理"""
    manager = get_cleanup_manager()
    manager.cleanup_shared_memory()


def force_cleanup():
    """强制清理所有相关的共享内存"""
    manager = get_cleanup_manager()
    manager.force_cleanup_by_pattern()


def cleanup_orphaned_shm():
    """清理孤儿共享内存段"""
    try:
        # 获取当前运行的LightLLM进程
        current_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['name'] and 'python' in proc.info['name']:
                    cmdline = proc.info['cmdline'] or []
                    if any('lightllm' in arg.lower() for arg in cmdline):
                        current_processes.append(proc.info['pid'])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        logger.info(f"Found {len(current_processes)} running LightLLM processes")
        
        # 如果没有运行的LightLLM进程，执行强制清理
        if not current_processes:
            logger.info("No running LightLLM processes found, performing force cleanup")
            force_cleanup()
        else:
            logger.info("LightLLM processes still running, skipping cleanup")
            
    except Exception as e:
        logger.error(f"Error in cleanup_orphaned_shm: {e}")


# 包装函数，在创建共享内存时自动注册清理
def create_shm_with_cleanup(name: str, size: int, create: bool = True) -> shared_memory.SharedMemory:
    """创建POSIX共享内存并自动注册清理"""
    try:
        if create:
            shm = shared_memory.SharedMemory(name=name, create=True, size=size)
            logger.info(f"Created POSIX shared memory {name}")
        else:
            shm = shared_memory.SharedMemory(name=name, create=False)
            logger.info(f"Attached to POSIX shared memory {name}")
        
        # 注册清理
        register_posix_shm_cleanup(name)
        return shm
        
    except FileExistsError:
        # 如果已存在，尝试连接
        shm = shared_memory.SharedMemory(name=name, create=False)
        logger.info(f"Attached to existing POSIX shared memory {name}")
        register_posix_shm_cleanup(name)
        return shm
    except Exception as e:
        logger.error(f"Failed to create/attach POSIX shared memory {name}: {e}")
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Shared Memory Cleanup Tool")
    parser.add_argument("--force", action="store_true", help="Force cleanup all shared memory")
    parser.add_argument("--orphaned", action="store_true", help="Cleanup orphaned shared memory")
    parser.add_argument("--key", type=int, help="Cleanup specific System V shared memory key")
    parser.add_argument("--name", type=str, help="Cleanup specific POSIX shared memory name")
    
    args = parser.parse_args()
    
    if args.force:
        force_cleanup()
    elif args.orphaned:
        cleanup_orphaned_shm()
    elif args.key:
        manager = get_cleanup_manager()
        manager.register_shm_key(args.key)
        manager.cleanup_shared_memory()
    elif args.name:
        manager = get_cleanup_manager()
        manager.register_posix_shm_name(args.name)
        manager.cleanup_shared_memory()
    else:
        manual_cleanup()
