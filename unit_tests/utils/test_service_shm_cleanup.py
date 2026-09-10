import ctypes
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from types import SimpleNamespace
import uuid

import psutil
import pytest

from lightllm.utils import service_shm_cleanup


def test_cleanup_service_shm_only_removes_matching_service(monkeypatch, tmp_path):
    shm_dir = tmp_path / "shm"
    shm_dir.mkdir()
    matching_names = ["service_0_req_pool", "service_0_token_load"]
    for name in [*matching_names, "service_1_req_pool", "other_service_0_value"]:
        (shm_dir / name).touch()
    monkeypatch.setattr(service_shm_cleanup, "SHM_DIR", shm_dir)

    service_shm_cleanup.ServiceShmCleanup.cleanup_posix_shm("service_0")

    assert all(not (shm_dir / name).exists() for name in matching_names)
    assert (shm_dir / "service_1_req_pool").exists()
    assert (shm_dir / "other_service_0_value").exists()


def test_cleanup_ipc_files_only_removes_matching_service(monkeypatch, tmp_path):
    service_socket = tmp_path / "_service_0_router"
    other_socket = tmp_path / "_service_1_router"
    sockets = []
    for path in [service_socket, other_socket]:
        sock = socket.socket(socket.AF_UNIX)
        sock.bind(str(path))
        sockets.append(sock)
    service_lock = tmp_path / "service_0_shm_port_args.lock"
    other_lock = tmp_path / "service_1_shm_port_args.lock"
    service_lock.touch()
    other_lock.touch()
    monkeypatch.setattr(service_shm_cleanup, "IPC_DIR", tmp_path)

    try:
        assert service_shm_cleanup.ServiceShmCleanup.cleanup_ipc_files("service_0") == 2
        assert not service_socket.exists()
        assert not service_lock.exists()
        assert other_socket.exists()
        assert other_lock.exists()
    finally:
        for sock in sockets:
            sock.close()


def test_system_v_shm_keys_follow_feature_switches(monkeypatch):
    monkeypatch.setenv(
        "LIGHTLLM_START_ARGS",
        json.dumps(
            {
                "run_mode": "prefill",
                "enable_cpu_cache": True,
                "enable_multimodal": False,
                "cpu_kv_cache_shm_id": 21,
                "multi_modal_cache_shm_id": 22,
            }
        ),
    )
    removed_system_v_keys = []
    monkeypatch.setattr(
        service_shm_cleanup.ServiceShmCleanup,
        "cleanup_posix_shm",
        staticmethod(lambda service_name: 0),
    )
    monkeypatch.setattr(
        service_shm_cleanup.ServiceShmCleanup,
        "cleanup_ipc_files",
        staticmethod(lambda service_name: 0),
    )
    monkeypatch.setattr(
        service_shm_cleanup.ServiceShmCleanup,
        "cleanup_system_v_shm",
        staticmethod(lambda keys: removed_system_v_keys.extend(keys) or len(keys)),
    )

    service_shm_cleanup.ServiceShmCleanup("current_service_0").cleanup_service_resources()

    assert removed_system_v_keys == [21]


def test_non_inference_mode_skips_system_v_shm_cleanup(monkeypatch):
    monkeypatch.setenv(
        "LIGHTLLM_START_ARGS",
        json.dumps(
            {
                "run_mode": "visual_only",
                "enable_cpu_cache": True,
                "enable_multimodal": True,
                "cpu_kv_cache_shm_id": 21,
                "multi_modal_cache_shm_id": 22,
            }
        ),
    )
    system_v_cleanup_calls = []
    monkeypatch.setattr(
        service_shm_cleanup.ServiceShmCleanup,
        "cleanup_posix_shm",
        staticmethod(lambda service_name: 0),
    )
    monkeypatch.setattr(
        service_shm_cleanup.ServiceShmCleanup,
        "cleanup_ipc_files",
        staticmethod(lambda service_name: 0),
    )
    monkeypatch.setattr(
        service_shm_cleanup.ServiceShmCleanup,
        "cleanup_system_v_shm",
        staticmethod(lambda keys: system_v_cleanup_calls.append(keys) or 0),
    )

    service_shm_cleanup.ServiceShmCleanup("current_service_0").cleanup_service_resources()

    assert system_v_cleanup_calls == []


def test_active_process_rejects_missing_pid_and_zombie(monkeypatch):
    monkeypatch.setattr(
        service_shm_cleanup.psutil,
        "Process",
        lambda pid: SimpleNamespace(status=lambda: psutil.STATUS_RUNNING),
    )
    assert service_shm_cleanup.is_process_active(123)
    monkeypatch.setattr(
        service_shm_cleanup.psutil,
        "Process",
        lambda pid: SimpleNamespace(status=lambda: psutil.STATUS_ZOMBIE),
    )
    assert not service_shm_cleanup.is_process_active(123)


def test_cleanup_process_checks_parent_every_two_seconds(monkeypatch):
    states = iter([object(), object(), None])
    sleeps = []
    cleanup_calls = []
    signal_calls = []
    monkeypatch.setattr(service_shm_cleanup, "is_process_active", lambda pid: next(states))
    monkeypatch.setattr(service_shm_cleanup.time, "sleep", sleeps.append)
    monkeypatch.setattr(service_shm_cleanup.signal, "signal", lambda sig, handler: signal_calls.append((sig, handler)))
    monkeypatch.setattr(
        service_shm_cleanup.ServiceShmCleanup,
        "cleanup_service_resources",
        lambda self: cleanup_calls.append(self.service_name),
    )

    service_shm_cleanup.run_launcher_shm_cleanup_process("service_0", 123)

    assert sleeps == [2.0, 2.0]
    assert cleanup_calls == ["service_0"]
    assert signal_calls == [
        (signal.SIGINT, signal.SIG_IGN),
        (signal.SIGTERM, signal.SIG_IGN),
        (signal.SIGHUP, signal.SIG_IGN),
    ]


def test_start_cleanup_process_is_independent_session(monkeypatch):
    popen_calls = []
    monkeypatch.setattr(service_shm_cleanup.os, "getpid", lambda: 123)
    monkeypatch.setattr(
        service_shm_cleanup.subprocess, "Popen", lambda *args, **kwargs: popen_calls.append((args, kwargs))
    )

    assert service_shm_cleanup.start_launcher_shm_cleanup_process("service_0") is None

    command = popen_calls[0][0][0]
    assert command[-2:] == ["service_0", "123"]
    assert popen_calls[0][1]["start_new_session"] is True
    assert "env" not in popen_calls[0][1]


LAUNCHER_CODE = r"""
import json, os, psutil, sys, time
from pathlib import Path
from lightllm.utils.service_shm_cleanup import start_launcher_shm_cleanup_process
service, directory, exit_mode = sys.argv[1:]
directory = Path(directory)
start_launcher_shm_cleanup_process(service)
watcher = psutil.Process().children()[0]
(Path('/dev/shm')/(service+'_req_pool')).write_bytes(b'test')
lock = Path('/tmp')/(service+'_shm_port_args.lock')
lock.touch()
(directory/'launcher.json').write_text(json.dumps({'watcher':watcher.pid}))
while not (directory/'exit').exists(): time.sleep(.01)
if exit_mode == 'normal': sys.exit(0)
if exit_mode == 'exception': raise RuntimeError('test launcher failure')
if exit_mode == 'os_exit': os._exit(7)
while True: time.sleep(1)
"""


def _wait_until(predicate, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    assert predicate(), "Timed out waiting for resource cleanup"


def _process_stopped(pid):
    try:
        return psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return True


def _cleanup_signals_are_ignored(pid):
    status = Path(f"/proc/{pid}/status").read_text()
    ignored = int(next(line.split()[1] for line in status.splitlines() if line.startswith("SigIgn:")), 16)
    return all(ignored & (1 << (sig.value - 1)) for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGHUP])


@pytest.mark.parametrize("exit_mode", ["normal", "exception", "os_exit", "SIGKILL", "SIGINT", "SIGTERM", "SIGHUP"])
def test_cleanup_process_after_real_launcher_exit(tmp_path, exit_mode):
    service = "cleanup_test_" + uuid.uuid4().hex
    libc = ctypes.CDLL(None)
    while True:
        key = uuid.uuid4().int % (2 ** 30) + 1
        shmid = libc.shmget(key, 4096, 0o1000 | 0o2000 | 0o600)
        if shmid >= 0:
            break
    environment = os.environ.copy()
    environment["LIGHTLLM_START_ARGS"] = json.dumps(
        {"run_mode": "normal", "enable_cpu_cache": True, "cpu_kv_cache_shm_id": key}
    )
    process = None
    watcher_pid = None
    try:
        process = subprocess.Popen(
            [sys.executable, "-c", LAUNCHER_CODE, service, str(tmp_path), exit_mode],
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        _wait_until(lambda: (tmp_path / "launcher.json").exists())
        watcher_pid = json.loads((tmp_path / "launcher.json").read_text())["watcher"]
        assert os.getsid(watcher_pid) == watcher_pid
        assert os.getsid(watcher_pid) != os.getsid(process.pid)
        _wait_until(lambda: _cleanup_signals_are_ignored(watcher_pid))
        assert (Path("/dev/shm") / (service + "_req_pool")).exists()
        assert libc.shmget(key, 0, 0) == shmid
        if exit_mode == "normal":
            os.kill(watcher_pid, signal.SIGTERM)
            time.sleep(0.1)
            assert not _process_stopped(watcher_pid)
        if exit_mode.startswith("SIG"):
            if exit_mode == "SIGKILL":
                process.kill()
            else:
                os.killpg(process.pid, getattr(signal, exit_mode))
        else:
            (tmp_path / "exit").touch()
        process.wait(timeout=10)
        _wait_until(lambda: _process_stopped(watcher_pid))
        assert not (Path("/dev/shm") / (service + "_req_pool")).exists()
        assert not (Path("/tmp") / (service + "_shm_port_args.lock")).exists()
        assert libc.shmget(key, 0, 0) == -1
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        if watcher_pid and not _process_stopped(watcher_pid):
            psutil.Process(watcher_pid).kill()
        libc.shmctl(shmid, 0, None)
        (Path("/dev/shm") / (service + "_req_pool")).unlink(missing_ok=True)
        (Path("/tmp") / (service + "_shm_port_args.lock")).unlink(missing_ok=True)
