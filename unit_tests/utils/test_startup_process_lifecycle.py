import ctypes
import importlib
import multiprocessing as mp
import os
import signal
import sys
import time
import traceback
from types import SimpleNamespace

import psutil
import pytest

from lightllm.utils import process_check, start_utils


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="requires Linux subreaper and fork")


def _kill_and_reap_children():
    for child in psutil.Process().children(recursive=True):
        try:
            os.kill(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    while True:
        try:
            os.waitpid(-1, 0)
        except ChildProcessError:
            break


def _run_metrics_lifecycle(failure_mode, result_writer):
    """Own all launcher descendants, including orphans, in an isolated process."""
    result = None
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
            raise OSError(ctypes.get_errno(), "could not become a child subreaper")

        mp.set_start_method("fork", force=True)
        ctx = mp.get_context("fork")
        events, event_writer = ctx.Pipe(duplex=False)
        metrics = importlib.import_module("lightllm.server.metrics.manager")
        envs_utils = importlib.import_module("lightllm.utils.envs_utils")

        # Keep the real metrics entry point and RPyC server, without monitor/shared-memory setup.
        metrics.Monitor = lambda args: SimpleNamespace()
        metrics.get_unique_server_name = lambda: "metrics-lifecycle-regression"
        metrics.get_shm_port_args = lambda: SimpleNamespace(metric_port=0)
        envs_utils.get_env_start_args = lambda: SimpleNamespace(enable_mps=False)
        # Speed up only the monitor's polling, without changing the shared time module.
        process_check.time = SimpleNamespace(sleep=lambda _: time.sleep(0.01))

        def failing_submodule(pipe_writer):
            event_writer.send(("failing", os.getpid()))
            pipe_writer.send("injected initialization failure")
            while True:
                time.sleep(100)

        def launcher_main():
            manager = start_utils.SubmoduleManager()
            metric_processes = manager.start_submodule_processes(
                start_funcs=[metrics.start_metric_manager],
                start_args=[(SimpleNamespace(metric_gateway=None, push_interval=1),)],
            )
            event_writer.send(("metrics", metric_processes[0].pid))
            if failure_mode == "parent-exit":
                # Bypass multiprocessing joins to exercise the metrics parent monitor itself.
                os._exit(2)
            manager.start_submodule_processes(start_funcs=[failing_submodule], start_args=[()])

        launcher = ctx.Process(target=launcher_main)
        launcher.start()
        reported = {}
        expected = {"metrics", "failing"} if failure_mode == "startup-failure" else {"metrics"}
        deadline = time.monotonic() + 8
        while set(reported) != expected:
            assert time.monotonic() < deadline, "launcher did not start the expected submodules"
            if events.poll(0.05):
                name, pid = events.recv()
                reported[name] = pid

        launcher.join(timeout=8)
        assert not launcher.is_alive(), "launcher hung while joining an earlier metrics process"
        assert launcher.exitcode == (1 if failure_mode == "startup-failure" else 2)

        if failure_mode == "parent-exit":
            # The real monitor must kill metrics after its launcher's immediate exit.
            deadline = time.monotonic() + 5
            while True:
                reaped_pid, status = os.waitpid(reported["metrics"], os.WNOHANG)
                if reaped_pid:
                    assert os.WIFSIGNALED(status)
                    assert os.WTERMSIG(status) == signal.SIGKILL
                    break
                assert time.monotonic() < deadline, "metrics did not detect its dead launcher"
                time.sleep(0.01)

        assert all(not psutil.pid_exists(pid) for pid in reported.values()), "submodule leaked after launcher exit"
        result = "ok"
    except BaseException:
        result = traceback.format_exc()
    finally:
        _kill_and_reap_children()
        result_writer.send(result)


@pytest.mark.parametrize("failure_mode", ["startup-failure", "parent-exit"])
def test_metrics_cleanup_on_launcher_failure(failure_mode):
    # Import before forking so framework imports happen only once per pytest run.
    importlib.import_module("lightllm.server.metrics.manager")
    ctx = mp.get_context("fork")
    result_reader, result_writer = ctx.Pipe(duplex=False)
    controller = ctx.Process(target=_run_metrics_lifecycle, args=(failure_mode, result_writer))
    controller.start()
    try:
        controller.join(timeout=20)
        assert not controller.is_alive(), "metrics lifecycle regression controller timed out"
        assert controller.exitcode == 0
        assert result_reader.poll(timeout=1), "metrics lifecycle controller returned no result"
        result = result_reader.recv()
        assert result == "ok", result
    finally:
        if controller.is_alive():
            for child in psutil.Process(controller.pid).children(recursive=True):
                try:
                    os.kill(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            controller.kill()
            controller.join(timeout=5)
        controller.close()
        result_reader.close()
        result_writer.close()
