"""由 launcher 统一管理 LightLLM 服务创建的共享内存。

设计背景
--------
LightLLM 的 router、model、HTTP server 等子进程通过共享内存交换状态。业务层使用逻辑名称，
共享内存基础层会统一添加 ``{service_name}_`` 前缀，因此同一台机器上的多个服务不会重名。
这些资源由 launcher 统一回收，子进程不单独注册信号处理函数。

支持的退出场景
--------------
1. SIGINT/SIGTERM/SIGHUP：launcher 先停止子进程，再主动调用本模块返回的 cleanup 函数。
2. Python 正常退出或未捕获异常：``atexit`` 作为兜底执行同一个 cleanup 函数。
3. 多实例并存：``/dev/shm`` 名称带有 service name，清理时只处理当前服务。

SIGKILL、OOM Killer 等场景无法执行 Python 清理回调，本模块不记录 owner 文件，也不在下次
启动时补偿清理这些异常残留。该功能定位为 launcher 有机会退出时执行的简单兜底清理。

所有启动模式都会创建 ShmPortArgs 等 POSIX 共享内存，因此统一按 service name 清理。System V
共享内存只可能由 normal、prefill、decode 推理节点创建，并继续按照 CPU KV Cache 和多模态缓存
功能开关选择有效 key；pd_master、visual_only、config_server 不执行 System V SHM 清理。

本模块只负责服务内部共享内存。由外部 RL 进程创建并通过协议传入完整名称的共享内存，不属于
当前 launcher 的服务前缀命名空间，因此不在这里按名称扫描回收。
"""

import atexit
import ctypes
import json
import os
import subprocess
from pathlib import Path

from lightllm.utils.log_utils import init_logger


logger = init_logger(__name__)

SHM_DIR = Path("/dev/shm")


class ServiceShmCleanup:
    """管理一个 launcher 在当前节点上拥有的共享内存。"""

    def __init__(self, service_name):
        if not service_name:
            raise RuntimeError("service_name must be initialized before registering shm cleanup")

        self.service_name = service_name

        try:
            self.start_args = json.loads(os.environ["LIGHTLLM_START_ARGS"])
        except (KeyError, json.JSONDecodeError):
            self.start_args = {}
        if not isinstance(self.start_args, dict):
            self.start_args = {}

    @staticmethod
    def cleanup_posix_shm(service_name):
        """删除名称严格属于目标 service 的 POSIX 共享内存。"""
        try:
            entries = [entry for entry in SHM_DIR.iterdir() if entry.name.startswith(f"{service_name}_")]
        except FileNotFoundError:
            return 0

        if not entries:
            return 0

        try:
            # 这是服务退出后的兜底路径：先按严格前缀筛选，再启动一次 rm 批量删除。
            # 使用参数列表和 "--"，避免 shell 管道、通配符展开及名称转义问题。
            subprocess.run(["rm", "-f", "--", *(str(entry) for entry in entries)], check=True)
        except (OSError, subprocess.CalledProcessError):
            logger.exception(f"Failed to remove POSIX shm for service {service_name}")
            return 0
        return len(entries)

    @staticmethod
    def cleanup_system_v_shm(keys):
        """删除启动参数中记录的 System V 共享内存。"""
        libc = ctypes.CDLL("/usr/lib/x86_64-linux-gnu/libc.so.6", use_errno=True)
        libc.shmget.argtypes = (ctypes.c_long, ctypes.c_size_t, ctypes.c_int)
        libc.shmget.restype = ctypes.c_int
        libc.shmctl.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_void_p)
        libc.shmctl.restype = ctypes.c_int

        removed = 0
        for key in keys:
            try:
                # shmget(key, size, shmflg)：这里只查找已有段，不创建新段；size=0 不申请空间，
                # shmflg=0 表示不附加 IPC_CREAT 等标志。成功返回非负的内核共享内存 ID，
                # 失败返回 -1。
                shmid = libc.shmget(int(key), 0, 0)
                if shmid < 0:
                    continue

                # shmctl(shmid, cmd, buf)：cmd=0 是 IPC_RMID，buf 在该命令下不使用，所以传 None。
                # IPC_RMID 将共享内存标记为删除；最后一个已 attach 的进程 detach 后才真正释放。
                # shmctl 成功返回 0，失败返回 -1。
                removed += int(libc.shmctl(shmid, 0, None) == 0)
            except Exception:
                logger.exception(f"Failed to remove System V shm key {key}")
        return removed

    def cleanup_service_resources(self):
        """回收当前服务实际创建的 POSIX 和 System V 共享内存。"""
        removed_posix = self.cleanup_posix_shm(self.service_name)
        system_v_shm_keys = []
        if self.start_args.get("run_mode") in ["normal", "prefill", "decode"]:
            if self.start_args.get("enable_cpu_cache") and self.start_args.get("cpu_kv_cache_shm_id") is not None:
                system_v_shm_keys.append(int(self.start_args["cpu_kv_cache_shm_id"]))
            if self.start_args.get("enable_multimodal") and self.start_args.get("multi_modal_cache_shm_id") is not None:
                system_v_shm_keys.append(int(self.start_args["multi_modal_cache_shm_id"]))
        removed_system_v = self.cleanup_system_v_shm(system_v_shm_keys) if system_v_shm_keys else 0
        if removed_posix or removed_system_v:
            logger.info(
                f"Cleaned service shm for {self.service_name}: "
                f"POSIX={removed_posix}, System V keys={removed_system_v}"
            )

    def register(self):
        """安装正常退出时的兜底清理回调。"""
        atexit.register(self.cleanup)
        return self.cleanup

    def cleanup(self):
        """执行幂等的兜底回收，允许主动退出流程和 atexit 重复调用。"""
        self.cleanup_service_resources()


def register_launcher_shm_cleanup(service_name):
    """创建 launcher 的清理对象，并返回 cleanup 函数。"""
    return ServiceShmCleanup(service_name).register()
