import ctypes
import importlib
import multiprocessing as mp
import os
import signal
import sys
import time
import traceback
from types import ModuleType, SimpleNamespace

import psutil
import pytest

from lightllm.utils import process_check


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


def _run_startup_failure(mode, late_monitor, result_connection):
    """Own orphaned workers in an isolated subreaper so the test can always reap them."""
    result = None
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
            raise OSError(ctypes.get_errno(), "could not become a child subreaper")

        ctx = mp.get_context("fork")
        events, event_writer = ctx.Pipe(duplex=False)
        start_monitor = ctx.Event()
        if not late_monitor:
            start_monitor.set()

        # Speed up the real monitor without changing the shared time module's sleep.
        process_check.time = SimpleNamespace(sleep=lambda _: time.sleep(0.01))

        def worker_main(started):
            if late_monitor:
                started.set()
            assert start_monitor.wait(timeout=10)
            event_writer.send(("monitor_parent", os.getpid(), os.getppid()))
            process_check.start_parent_check_thread()
            started.set()
            while True:
                time.sleep(100)

        def start_worker(*args, **kwargs):
            started = ctx.Event()
            worker = ctx.Process(target=worker_main, args=(started,))
            worker.start()
            event_writer.send(("worker", worker.pid))
            assert not worker.daemon
            assert started.wait(timeout=5)

        module_name = "lightllm.server.router.model_infer.mode_backend.pd." f"{mode}_node_impl.{mode}_kv_move_manager"
        manager_module = importlib.import_module(module_name)
        manager_module.get_unique_server_name = lambda: "startup-regression"
        original_logger = manager_module.logger

        def log_failure(message):
            original_logger.exception(message)
            event_writer.send(("init_error",))

        manager_module.logger = SimpleNamespace(exception=log_failure)

        def fail_initialization(*args, **kwargs):
            start_worker()
            raise RuntimeError("injected transport initialization failure")

        manager_class = "PrefillKVMoveManager" if mode == "prefill" else "DecodeKVMoveManager"
        setattr(manager_module, manager_class, fail_initialization)

        # Exercise the real entry point while avoiding CUDA and network initialization.
        package_name = module_name.rsplit(".", 1)[0]
        transport = ModuleType(f"{package_name}.{mode}_trans_process")
        setattr(transport, f"start_{mode}_trans_process", lambda *args, **kwargs: None)
        sys.modules[transport.__name__] = transport
        sys.modules["lightllm.utils.rpyc_fix_utils"] = ModuleType("lightllm.utils.rpyc_fix_utils")
        if mode == "decode":
            up_status = ModuleType(f"{package_name}.up_status")
            up_status.start_up_kv_status_process = start_worker
            sys.modules[up_status.__name__] = up_status

        ready = ctx.Event()
        manager = ctx.Process(target=manager_module._init_env, args=(SimpleNamespace(), None, ready))
        manager.start()
        manager.join(timeout=5)
        assert not manager.is_alive(), "manager hung while joining its non-daemon children"
        assert manager.exitcode == 1
        assert not ready.is_set()

        # In the late case workers are already reparented when they start the monitor.
        start_monitor.set()
        worker_count = 2 if mode == "decode" else 1
        workers = set()
        monitor_parents = {}
        logged_failure = False
        deadline = time.monotonic() + 5
        while len(workers) < worker_count or len(monitor_parents) < worker_count or not logged_failure:
            assert time.monotonic() < deadline, "missing worker or initialization failure report"
            if not events.poll(0.05):
                continue
            event = events.recv()
            if event[0] == "worker":
                workers.add(event[1])
            elif event[0] == "monitor_parent":
                monitor_parents[event[1]] = event[2]
            elif event[0] == "init_error":
                logged_failure = True

        expected_parent = os.getpid() if late_monitor else manager.pid
        assert set(monitor_parents) == workers
        assert set(monitor_parents.values()) == {expected_parent}

        for pid in workers:
            while True:
                reaped_pid, status = os.waitpid(pid, os.WNOHANG)
                if reaped_pid:
                    assert os.WIFSIGNALED(status)
                    assert os.WTERMSIG(status) == signal.SIGKILL
                    break
                assert time.monotonic() < deadline, "worker did not detect its dead parent"
                time.sleep(0.01)
        result = "ok"
    except BaseException:
        result = traceback.format_exc()
    finally:
        _kill_and_reap_children()
        result_connection.send(result)


@pytest.mark.parametrize("mode", ["prefill", "decode"])
@pytest.mark.parametrize("late_monitor", [False, True], ids=["monitor-running", "monitor-after-parent-exit"])
def test_failed_manager_exits_and_workers_detect_parent_death(mode, late_monitor):
    # Import before forking so costly framework imports happen only once per test run.
    importlib.import_module(
        f"lightllm.server.router.model_infer.mode_backend.pd.{mode}_node_impl.{mode}_kv_move_manager"
    )
    ctx = mp.get_context("fork")
    result_reader, result_writer = ctx.Pipe(duplex=False)
    controller = ctx.Process(target=_run_startup_failure, args=(mode, late_monitor, result_writer))
    controller.start()
    try:
        controller.join(timeout=15)
        assert not controller.is_alive(), "startup regression controller timed out"
        assert controller.exitcode == 0
        assert result_reader.poll(timeout=1), "startup regression controller returned no result"
        assert result_reader.recv() == "ok"
    finally:
        if controller.is_alive():
            # The controller normally owns cleanup; retain a bounded test-runner fallback.
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
