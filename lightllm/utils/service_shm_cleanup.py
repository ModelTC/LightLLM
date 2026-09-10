import atexit
import ctypes
import json
import os
from multiprocessing import shared_memory
from pathlib import Path

import psutil
from filelock import FileLock

from lightllm.utils.log_utils import init_logger


logger = init_logger(__name__)

SHM_DIR = Path("/dev/shm")
OWNER_DIR = Path("/tmp/lightllm_service_owners")
OWNER_LOCK_PATH = Path("/tmp/lightllm_service_owners.lock")
SYSTEM_V_SHM_KEY_NAMES = ("cpu_kv_cache_shm_id", "multi_modal_cache_shm_id")

_registered_cleanups = {}


def _get_system_v_shm_keys():
    try:
        start_args = json.loads(os.environ["LIGHTLLM_START_ARGS"])
    except (KeyError, json.JSONDecodeError):
        return []
    return [int(start_args[name]) for name in SYSTEM_V_SHM_KEY_NAMES if start_args.get(name) is not None]


def _unlink_posix_shm(name):
    # name 来自 /dev/shm 的实际条目，可能属于上一次异常退出的其他 service。
    # 这里必须直接使用完整名称，不能再按当前 service 添加前缀。
    shm = shared_memory.SharedMemory(name=name, create=False)
    try:
        shm.unlink()
    finally:
        shm.close()


def _remove_system_v_shm(key):
    libc = ctypes.CDLL("/usr/lib/x86_64-linux-gnu/libc.so.6", use_errno=True)
    libc.shmget.argtypes = (ctypes.c_long, ctypes.c_size_t, ctypes.c_int)
    libc.shmget.restype = ctypes.c_int
    libc.shmctl.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_void_p)
    libc.shmctl.restype = ctypes.c_int

    shmid = libc.shmget(int(key), 0, 0)
    if shmid >= 0:
        return libc.shmctl(shmid, 0, None) == 0  # IPC_RMID
    return False


def cleanup_service_shm(service_name, system_v_shm_keys=()):
    """回收一个 LightLLM 服务在本机创建的共享内存。"""
    if not service_name:
        return

    prefix = f"{service_name}_"
    removed_posix = 0
    try:
        entries = list(SHM_DIR.iterdir())
    except FileNotFoundError:
        entries = []

    for entry in entries:
        if not entry.name.startswith(prefix):
            continue
        try:
            _unlink_posix_shm(entry.name)
            removed_posix += 1
        except FileNotFoundError:
            pass
        except Exception:
            logger.exception(f"Failed to unlink POSIX shm {entry.name}")

    removed_system_v = 0
    for key in system_v_shm_keys:
        try:
            removed_system_v += int(_remove_system_v_shm(key))
        except Exception:
            logger.exception(f"Failed to remove System V shm key {key}")

    if removed_posix or removed_system_v:
        logger.info(
            f"Cleaned service shm for {service_name}: " f"POSIX={removed_posix}, System V keys={removed_system_v}"
        )


def _owner_is_alive(owner):
    try:
        process = psutil.Process(int(owner["pid"]))
        return (
            process.is_running()
            and process.status() != psutil.STATUS_ZOMBIE
            and abs(process.create_time() - float(owner["create_time"])) < 0.01
        )
    except psutil.AccessDenied:
        # 无权限确认的进程按存活处理，避免清理其他用户正在运行的服务。
        return True
    except (KeyError, TypeError, ValueError, psutil.NoSuchProcess, psutil.ZombieProcess):
        return False


def _load_owner(path):
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError):
        logger.warning(f"Ignore invalid LightLLM service owner file: {path}")
        return None


def _recover_stale_services():
    for owner_path in OWNER_DIR.glob("*.json"):
        owner = _load_owner(owner_path)
        if owner is None or _owner_is_alive(owner):
            continue
        cleanup_service_shm(owner.get("service_name"), owner.get("system_v_shm_keys", ()))
        try:
            owner_path.unlink()
        except FileNotFoundError:
            pass


def register_launcher_shm_cleanup(service_name):
    """登记 launcher 资源，并在退出或下次启动时回收该服务的共享内存。"""
    if not service_name:
        raise RuntimeError("service_name must be initialized before registering shm cleanup")

    owner_pid = os.getpid()
    cleanup_key = (owner_pid, service_name)
    if cleanup_key in _registered_cleanups:
        return _registered_cleanups[cleanup_key]

    OWNER_DIR.mkdir(parents=True, exist_ok=True)
    system_v_shm_keys = _get_system_v_shm_keys()
    owner = {
        "service_name": service_name,
        "pid": owner_pid,
        "create_time": psutil.Process(owner_pid).create_time(),
        "system_v_shm_keys": system_v_shm_keys,
    }
    owner_path = OWNER_DIR / f"{service_name}.json"

    with FileLock(str(OWNER_LOCK_PATH)):
        _recover_stale_services()
        tmp_path = owner_path.with_suffix(f".{owner_pid}.tmp")
        with tmp_path.open("w", encoding="utf-8") as file:
            json.dump(owner, file)
        os.replace(tmp_path, owner_path)

    def cleanup():
        nonlocal cleaned
        # multiprocessing 子进程会继承 atexit 回调，只有创建记录的 launcher 可以执行全量回收。
        if cleaned or os.getpid() != owner_pid:
            return
        cleaned = True
        cleanup_service_shm(service_name, system_v_shm_keys)
        with FileLock(str(OWNER_LOCK_PATH)):
            current_owner = _load_owner(owner_path)
            if current_owner is not None and current_owner.get("pid") == owner_pid:
                try:
                    owner_path.unlink()
                except FileNotFoundError:
                    pass

    cleaned = False
    _registered_cleanups[cleanup_key] = cleanup
    atexit.register(cleanup)
    return cleanup
