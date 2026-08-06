"""
ShmPortArgs: 跨进程共享的启动端口表（Shared Memory Port Args）。

设计目的
--------
LightLLM 启动时有两类端口：
  [user_set] 用户通过 CLI / 默认值已经写在 start args 里（如 --port、--pd_master_port）。
             未设置时访问会直接报错，不会动态分配。
  [dynamic]  args 中为 None，需要运行时动态申请（如 router_port、metric_port）。

本类把这两类端口统一成属性访问接口，并在进程间通过命名 POSIX shm 共享
「已动态分配」的结果，避免启动阶段预先批量占坑，也避免多进程重复分配冲突。

存储与并发
----------
- shm 名：`{unique_server_name}_shm_port_args`
- 内容：pickle 序列化的 dict[str, int | list[int]]
- 分配：socket.bind("", 0) 取空闲端口；分配前排除
    1) start args 中已设置的 port 字段
    2) 本 shm 中已经分配过的端口
- 互斥：FileLock(`/tmp/{shm_name}.lock`)

使用约定
--------
使用 ShmPortArgs 之前，必须先初始化以下环境信息（否则会直接抛错）：
  1. `set_unique_server_name(args)`
     → 写入 `LIGHTLLM_UNIQUE_SERVICE_NAME_ID`，供 `get_unique_server_name()` 使用，用于拼 shm 名。
  2. `set_env_start_args(args)`
     → 写入 `LIGHTLLM_START_ARGS`，供 `get_env_start_args()` 使用，用于读取用户已设置端口、
       以及 visual_dp / audio_dp 等分配参数。

之后：
  3. 启动主进程：`ShmPortArgs.get_instance(create=True)` 创建空表。
  4. 同进程其它位置 / 子进程：`ShmPortArgs.get_instance()` 复用或 link 同一块 shm。
  5. 请通过 `get_instance` / `get_shm_port_args` 获取对象；直接构造会绕过进程内单例。

示例
----
    set_unique_server_name(args)   # 初始化 get_unique_server_name 所需环境变量
    set_env_start_args(args)       # 初始化 get_env_start_args 所需环境变量

    ShmPortArgs.get_instance(create=True)
    ports = get_shm_port_args()
    router_port = ports.router_port          # dynamic：按需分配
    http_port = ports.port                  # user_set：读 args
    vit_ports = ports.visual_nccl_ports     # dynamic list
"""

from __future__ import annotations

import os
import pickle
import socket
import struct
from typing import Dict, List, Set, Union

from filelock import FileLock

from lightllm.utils.envs_utils import get_env_start_args, get_unique_server_name
from lightllm.utils.log_utils import init_logger
from lightllm.utils.shm_utils import create_or_link_shm

logger = init_logger(__name__)

PortValue = Union[int, List[int]]


class ShmPortArgs:
    _SHM_SIZE = 64 * 1024
    _ALLOC_MAX_RETRY = 128
    _instance: "ShmPortArgs | None" = None

    def __init__(self, create: bool = False):
        uni = get_unique_server_name()
        if not uni:
            raise RuntimeError(
                "LIGHTLLM_UNIQUE_SERVICE_NAME_ID is unset; " "call set_unique_server_name(args) before ShmPortArgs"
            )
        if "LIGHTLLM_START_ARGS" not in os.environ:
            raise RuntimeError("LIGHTLLM_START_ARGS is unset; call set_env_start_args(args) before ShmPortArgs")

        self._shm_name = f"{uni}_shm_port_args"
        self._lock = FileLock(f"/tmp/{self._shm_name}.lock")
        self.shm = create_or_link_shm(
            self._shm_name,
            self._SHM_SIZE,
            force_mode="create" if create else "link",
            auto_cleanup=create,
        )
        if create:
            self._save({})

    @classmethod
    def get_instance(cls, create: bool = False) -> "ShmPortArgs":
        """同进程单例；create 仅在首次构造时生效。"""
        if cls._instance is None:
            cls._instance = cls(create=create)
        return cls._instance

    # =====================================================================
    # [user_set] 用户已在 args 中设置（含 CLI 默认值）；未设置则报错
    # =====================================================================

    # [user_set] HTTP API listen port
    @property
    def port(self) -> int:
        return self._get_user_set_port("port")

    # [user_set] PD master port
    @property
    def pd_master_port(self) -> int:
        return self._get_user_set_port("pd_master_port")

    # [user_set] config server port
    @property
    def config_server_port(self) -> int:
        return self._get_user_set_port("config_server_port")

    # [user_set] config server visual redis port
    @property
    def config_server_visual_redis_port(self) -> int:
        return self._get_user_set_port("config_server_visual_redis_port")

    # [user_set] multinode http manager port
    @property
    def multinode_httpmanager_port(self) -> int:
        return self._get_user_set_port("multinode_httpmanager_port")

    # [user_set] multinode router gloo port
    @property
    def multinode_router_gloo_port(self) -> int:
        return self._get_user_set_port("multinode_router_gloo_port")

    # =====================================================================
    # [dynamic] args 可为 None，需要时动态分配并写入 shm
    # =====================================================================

    # [dynamic] pytorch distributed / NCCL TCPStore port
    @property
    def nccl_port(self) -> int:
        return self._get_from_args_or_alloc("nccl_port")

    # [dynamic] visual-only RPyC port
    @property
    def visual_rpyc_port(self) -> int:
        return self._get_from_args_or_alloc("visual_rpyc_port")

    # [dynamic] visual vit nccl ports (list, len=visual_dp)
    @property
    def visual_nccl_ports(self) -> List[int]:
        ports = self._get_from_args_or_alloc("visual_nccl_ports", count=int(get_env_start_args().visual_dp))
        return [ports] if isinstance(ports, int) else ports

    # [dynamic] audio encoder nccl ports (list, len=audio_dp)
    @property
    def audio_nccl_ports(self) -> List[int]:
        ports = self._get_from_args_or_alloc("audio_nccl_ports", count=int(get_env_start_args().audio_dp))
        return [ports] if isinstance(ports, int) else ports

    # [dynamic] router zmq port
    @property
    def router_port(self) -> int:
        return self._get_from_args_or_alloc("router_port")

    # [dynamic] router profiler zmq port
    @property
    def router_profiler_port(self) -> int:
        return self._get_from_args_or_alloc("router_profiler_port")

    # [dynamic] detokenization zmq port
    @property
    def detokenization_port(self) -> int:
        return self._get_from_args_or_alloc("detokenization_port")

    # [dynamic] http server internal zmq port
    @property
    def http_server_port(self) -> int:
        return self._get_from_args_or_alloc("http_server_port")

    # [dynamic] visual server zmq port
    @property
    def visual_port(self) -> int:
        return self._get_from_args_or_alloc("visual_port")

    # [dynamic] audio server zmq port
    @property
    def audio_port(self) -> int:
        return self._get_from_args_or_alloc("audio_port")

    # [dynamic] embed cache rpyc port
    @property
    def cache_port(self) -> int:
        return self._get_from_args_or_alloc("cache_port")

    # [dynamic] metrics rpyc port
    @property
    def metric_port(self) -> int:
        return self._get_from_args_or_alloc("metric_port")

    # [dynamic] multi-level kv cache port
    @property
    def multi_level_kv_cache_port(self) -> int:
        return self._get_from_args_or_alloc("multi_level_kv_cache_port")

    # [dynamic] router RL RPyC port
    @property
    def rl_rpyc_port(self) -> int:
        return self._get_from_args_or_alloc("rl_rpyc_port")

    def close(self) -> None:
        if self.shm is not None:
            self.shm.close()
            self.shm = None
        if ShmPortArgs._instance is self:
            ShmPortArgs._instance = None

    def _get_user_set_port(self, name: str) -> int:
        """只读 args；未设置则报错，不动态分配。"""
        value = getattr(get_env_start_args(), name, None)
        if value is None:
            raise RuntimeError(f"user_set port '{name}' is None; set it in start args before use")
        return int(value)

    def _get_from_args_or_alloc(self, name: str, count: int = 1) -> PortValue:
        """args 已设置则用用户值，否则写入同一份 shm 动态分配。"""
        value = getattr(get_env_start_args(), name, None)
        if value is not None:
            if count == 1:
                return int(value)
            return [int(v) for v in value[:count]]
        return self._get_or_alloc(name, count=count)

    def _get_or_alloc(self, name: str, count: int = 1) -> PortValue:
        with self._lock:
            ports = self._load()
            if name in ports:
                return ports[name]

            # 分配时必须同时排除：
            # 1) args 中用户已设置的端口
            # 2) 本 shm 中已经分配过的端口
            reserved = self._ports_from_args() | self._ports_from_shm(ports)

            if count == 1:
                port = self._alloc_free_port(reserved)
                ports[name] = port
                self._save(ports)
                logger.info(f"ShmPortArgs alloc {name}={port}")
                return port

            allocated: List[int] = []
            for _ in range(count):
                port = self._alloc_free_port(reserved)
                reserved.add(port)  # 本轮后续分配也要排除刚分到的端口
                allocated.append(port)
            ports[name] = allocated
            self._save(ports)
            logger.info(f"ShmPortArgs alloc {name}={allocated}")
            return allocated

    @staticmethod
    def _ports_from_args() -> Set[int]:
        """遍历 args，收集 key 含 port 且 value 非 None 的端口，分配时必须避开。"""
        args = get_env_start_args()
        reserved: Set[int] = set()
        for key, value in dict(args).items():
            if "port" not in key.lower() or value is None:
                continue
            if isinstance(value, (list, tuple)):
                reserved.update(int(v) for v in value if v is not None)
            else:
                reserved.add(int(value))
        return reserved

    @staticmethod
    def _ports_from_shm(ports: Dict[str, PortValue]) -> Set[int]:
        """收集 shm 中已经分配过的端口。"""
        reserved: Set[int] = set()
        for value in ports.values():
            if isinstance(value, list):
                reserved.update(int(v) for v in value)
            else:
                reserved.add(int(value))
        return reserved

    def _load(self) -> Dict[str, PortValue]:
        n = struct.unpack_from("I", self.shm.buf, 0)[0]
        if n == 0:
            return {}
        return pickle.loads(bytes(self.shm.buf[4 : 4 + n]))

    def _save(self, ports: Dict[str, PortValue]) -> None:
        blob = pickle.dumps(ports, protocol=pickle.HIGHEST_PROTOCOL)
        if 4 + len(blob) > self._SHM_SIZE:
            raise RuntimeError(f"port table too large: {len(blob)} bytes")
        struct.pack_into("I", self.shm.buf, 0, len(blob))
        self.shm.buf[4 : 4 + len(blob)] = blob

    def _alloc_free_port(self, reserved: Set[int]) -> int:
        for _ in range(self._ALLOC_MAX_RETRY):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(("", 0))
                port = int(sock.getsockname()[1])
            if port in reserved:
                logger.warning(f"skip port {port}, already in args or shm-allocated")
                continue
            if not self._is_port_free(port):
                logger.warning(f"skip port {port}, not free")
                continue
            return port
        raise RuntimeError(f"failed to allocate free port after {self._ALLOC_MAX_RETRY} retries, reserved={reserved}")

    @staticmethod
    def _is_port_free(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("", port))
                return True
            except OSError:
                return False


def get_shm_port_args(create: bool = False) -> ShmPortArgs:
    """Convenience accessor for the process-local ShmPortArgs singleton."""
    return ShmPortArgs.get_instance(create=create)
