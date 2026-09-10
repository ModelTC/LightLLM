"""由 launcher 统一管理 LightLLM 服务创建的共享内存。

设计背景
--------
LightLLM 的 router、model、HTTP server 等子进程通过共享内存交换状态。业务层使用逻辑名称，
共享内存基础层会统一添加 ``{service_name}_`` 前缀，因此同一台机器上的多个服务不会重名。
这些资源由 launcher 统一回收，子进程不单独注册信号处理函数。

支持的退出场景
--------------
1. 独立清理进程每 2 秒检查 launcher；正常退出、异常退出或 SIGKILL 后统一回收。
2. SIGINT/SIGTERM/SIGHUP：launcher 先停止子进程，退出后由清理进程回收资源。
3. 多实例并存：``/dev/shm`` 名称带有 service name，清理时只处理当前服务。

清理进程使用独立会话，不参与 launcher 的 multiprocessing 退出等待，也不会随终端进程组
信号一起退出。清理进程自身也被杀死或整个容器/机器退出的场景仍需要外部管理器负责回收。

所有启动模式都会创建 ShmPortArgs 等 POSIX 共享内存，因此统一按 service name 清理。System V
共享内存只可能由 normal、prefill、decode 推理节点创建，并继续按照 CPU KV Cache 和多模态缓存
功能开关选择有效 key；pd_master、visual_only、config_server 不执行 System V SHM 清理。

本模块只负责服务内部共享内存。由外部 RL 进程创建并通过协议传入完整名称的共享内存，不属于
当前 launcher 的服务前缀命名空间，因此不在这里按名称扫描回收。
"""

import ctypes
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil

from lightllm.utils.log_utils import init_logger


logger = init_logger(__name__)

SHM_DIR = Path("/dev/shm")
IPC_DIR = Path("/tmp")
PARENT_CHECK_INTERVAL = 2.0


class ServiceShmCleanup:
    """管理一个 launcher 在当前节点上拥有的共享内存。"""

    def __init__(self, service_name):
        if not service_name:
            raise RuntimeError("service_name must be initialized before starting shm cleanup")

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
    def cleanup_ipc_files(service_name):
        """删除当前 service 的 ZMQ IPC socket 和端口表锁文件。"""
        removed = 0
        for entry in IPC_DIR.iterdir():
            if (entry.name.startswith(f"_{service_name}_") and entry.is_socket()) or entry.name == (
                f"{service_name}_shm_port_args.lock"
            ):
                entry.unlink(missing_ok=True)
                removed += 1
        return removed

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
        """回收当前服务的 POSIX/System V 共享内存及 IPC 文件。"""
        removed_posix = self.cleanup_posix_shm(self.service_name)
        system_v_shm_keys = []
        if self.start_args.get("run_mode") in ["normal", "prefill", "decode"]:
            if self.start_args.get("enable_cpu_cache") and self.start_args.get("cpu_kv_cache_shm_id") is not None:
                system_v_shm_keys.append(int(self.start_args["cpu_kv_cache_shm_id"]))
            if self.start_args.get("enable_multimodal") and self.start_args.get("multi_modal_cache_shm_id") is not None:
                system_v_shm_keys.append(int(self.start_args["multi_modal_cache_shm_id"]))
        removed_system_v = self.cleanup_system_v_shm(system_v_shm_keys) if system_v_shm_keys else 0
        removed_ipc = self.cleanup_ipc_files(self.service_name)
        if removed_posix or removed_system_v or removed_ipc:
            logger.info(
                f"Cleaned service shm for {self.service_name}: "
                f"POSIX={removed_posix}, System V keys={removed_system_v}, IPC files={removed_ipc}"
            )


def is_process_active(pid):
    """检查进程是否存在且不是僵尸进程。"""
    try:
        process = psutil.Process(pid)
        return process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def run_launcher_shm_cleanup_process(service_name, parent_pid):
    """每 2 秒检查 launcher，launcher 退出后清理其服务资源。"""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    while is_process_active(parent_pid):
        time.sleep(PARENT_CHECK_INTERVAL)
    logger.info(f"Launcher {parent_pid} exited; cleaning service {service_name}")
    ServiceShmCleanup(service_name).cleanup_service_resources()


def start_launcher_shm_cleanup_process(service_name):
    """启动独立于 launcher 进程组的资源清理进程。"""
    cleanup_process_code = (
        "import sys; "
        "from lightllm.utils.service_shm_cleanup import run_launcher_shm_cleanup_process; "
        "run_launcher_shm_cleanup_process(sys.argv[1], int(sys.argv[2]))"
    )
    subprocess.Popen(
        [
            sys.executable,
            "-c",
            cleanup_process_code,
            service_name,
            str(os.getpid()),
        ],
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
