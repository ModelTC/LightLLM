import errno
import multiprocessing as mp
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest
import zmq

from lightllm.server.multi_level_kv_cache import manager as manager_impl
from lightllm.utils.start_utils import SubmoduleManager


def _start_cache(stage, commands, pipe_writer):
    context = Mock()
    send_socket = MagicMock()
    if stage == "connect":
        send_socket.connect.side_effect = zmq.ZMQError(errno.EMFILE)
    send_socket.send_pyobj.side_effect = zmq.ZMQError(errno.EIO)

    def socket(kind):
        if kind == zmq.PUSH:
            if stage == "socket":
                raise zmq.ZMQError(errno.EMFILE)
            return send_socket
        return Mock()

    def task_loop(self):
        command = commands.recv()
        if command == "send":
            self.send_to_router_queue.put(None)
            self._send_finished_groups()
            commands.recv()

    context.socket.side_effect = socket
    context_factory = manager_impl.zmq.Context if stage == "ready" else lambda _: context

    with patch.multiple(
        manager_impl,
        CpuKvCacheClient=Mock(),
        ShmReqManager=Mock(),
        get_shm_port_args=lambda: SimpleNamespace(multi_level_kv_cache_port=1, router_port=2),
        get_unique_server_name=lambda: "cache-startup-test",
        start_parent_check_thread=lambda: None,
    ), patch.multiple(
        manager_impl.MultiLevelKVCacheManager,
        cpu_cache_hanle_loop=lambda self: None,
        task_loop=task_loop,
    ), patch.object(
        manager_impl.zmq, "Context", context_factory
    ):
        manager_impl.start_multi_level_kv_cache_manager(
            SimpleNamespace(
                zmq_mode="inproc://",
                enable_cpu_cache=True,
                enable_disk_cache=False,
                running_max_req_size=2,
            ),
            pipe_writer,
        )


def _run_launcher(stage):
    mp.set_start_method("spawn")
    commands, controller = mp.Pipe(duplex=False)
    try:
        SubmoduleManager().start_submodule_processes(start_funcs=[_start_cache], start_args=[(stage, commands)])
        assert stage in ("ready", "send")
        child = mp.active_children()[0]
        controller.send("send" if stage == "send" else "stop")
        child.join(timeout=5)
        assert child.exitcode == (1 if stage == "send" else 0)
    finally:
        for child in mp.active_children():
            child.kill()
            child.join(timeout=5)
        commands.close()
        controller.close()


@pytest.mark.parametrize("stage", ["socket", "connect", "ready", "send"])
def test_sender_startup(stage):
    result = subprocess.run(
        [sys.executable, "-m", "unit_tests.server.multi_level_kv_cache.test_manager_startup", stage],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        timeout=30,
    )
    output = result.stdout + result.stderr
    if stage in ("socket", "connect"):
        assert result.returncode == 1, output
        assert "init func _start_cache : Too many open files" in output, output
        assert "init func _start_cache : init ok" not in output, output
    else:
        assert result.returncode == 0, output
        assert "init func _start_cache : init ok" in output, output
        if stage == "send":
            assert "Input/output error" in output, output


if __name__ == "__main__":
    _run_launcher(sys.argv[1])
