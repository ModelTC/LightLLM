from types import SimpleNamespace

import pytest

from lightllm.utils import start_utils


class FakeHttpServerProcess:
    def __init__(self, return_code=None):
        self.return_code = return_code
        self.pid = 4321
        self.sent_signals = []
        self.wait_timeouts = []

    def poll(self):
        return self.return_code

    def send_signal(self, sig):
        self.sent_signals.append(sig)

    def wait(self, timeout=None):
        self.wait_timeouts.append(timeout)
        return self.return_code


class FakeProcess:
    def __init__(self, pid, running=True, children=None, name="process", exitcode=None, wait_timeout=False):
        self.pid = pid
        self.running = running
        self._children = children or []
        self._name = name
        self.exitcode = exitcode
        self.wait_timeout = wait_timeout
        self.kill_calls = 0

    def is_running(self):
        return self.running

    def children(self, recursive=False):
        return self._children

    def name(self):
        return self._name

    def wait(self, timeout=None):
        if self.wait_timeout:
            raise start_utils.psutil.TimeoutExpired(timeout, pid=self.pid, name=self._name)
        return self.exitcode

    def kill(self):
        self.kill_calls += 1


def test_start_submodule_processes_returns_and_manages_psutil_processes(monkeypatch):
    class FakePipeReader:
        def recv(self):
            return "init ok"

    class FakeMpProcess:
        next_pid = 1000

        def __init__(self, target, args):
            self.pid = self.next_pid
            FakeMpProcess.next_pid += 1

        def start(self):
            pass

        def is_alive(self):
            return True

    monkeypatch.setattr(start_utils.mp, "Pipe", lambda duplex: (FakePipeReader(), object()))
    monkeypatch.setattr(start_utils.mp, "Process", FakeMpProcess)
    monkeypatch.setattr(
        start_utils.psutil,
        "Process",
        lambda pid: FakeProcess(pid=pid, name=f"process-{pid}"),
    )
    process_manager = start_utils.SubmoduleManager()

    processes = process_manager.start_submodule_processes(
        start_funcs=[lambda pipe_writer: None, lambda pipe_writer: None],
        start_args=[(), ()],
    )

    assert processes == process_manager.processes
    assert [process.pid for process in processes] == [1000, 1001]
    assert process_manager.process_names == {
        processes[0]: "process-1000",
        processes[1]: "process-1001",
    }


def test_start_submodule_processes_cleans_up_after_recv_error(monkeypatch):
    class FakePipeReader:
        def recv(self):
            raise EOFError

    class FakeMpProcess:
        pid = 1000

        def __init__(self, target, args):
            pass

        def start(self):
            pass

    managed_process = FakeProcess(pid=1000, name="process-1000")
    process_manager = start_utils.SubmoduleManager()
    cleanup_process_pids = []
    monkeypatch.setattr(start_utils.mp, "Pipe", lambda duplex: (FakePipeReader(), object()))
    monkeypatch.setattr(start_utils.mp, "Process", FakeMpProcess)
    monkeypatch.setattr(start_utils.psutil, "Process", lambda pid: managed_process)
    monkeypatch.setattr(
        process_manager,
        "terminate_all_processes",
        lambda: cleanup_process_pids.extend(process.pid for process in process_manager.processes),
    )

    with pytest.raises(EOFError):
        process_manager.start_submodule_processes(start_funcs=[lambda pipe_writer: None], start_args=[()])

    assert cleanup_process_pids == [1000]


def test_register_process_tree_adds_recursive_descendants(monkeypatch):
    descendants = [
        FakeProcess(pid=1001, name="lightllm::model_infer"),
        FakeProcess(pid=1002, name="lightllm::pd_manager"),
        FakeProcess(pid=1003, name="lightllm::pd_worker"),
    ]
    router_process = FakeProcess(pid=1000, children=descendants)
    process_manager = start_utils.SubmoduleManager()
    monkeypatch.setattr(start_utils, "is_process_active", lambda pid: True)

    process_manager.register_process_tree(router_process)

    assert process_manager.processes == descendants
    assert process_manager.process_names == {
        descendants[0]: "lightllm::model_infer",
        descendants[1]: "lightllm::pd_manager",
        descendants[2]: "lightllm::pd_worker",
    }


def test_register_process_tree_filters_short_lived_helper_processes(monkeypatch):
    model_process = FakeProcess(pid=1001, name="lightllm::model_infer")
    compile_worker = FakeProcess(pid=1002, name="python")
    pd_process = FakeProcess(pid=1003, name="lightllm::decode_trans")
    router_process = FakeProcess(pid=1000, children=[model_process, compile_worker, pd_process])
    process_manager = start_utils.SubmoduleManager()
    monkeypatch.setattr(start_utils, "is_process_active", lambda pid: True)

    process_manager.register_process_tree(router_process)

    assert process_manager.processes == [model_process, pd_process]
    assert process_manager.process_names == {
        model_process: "lightllm::model_infer",
        pd_process: "lightllm::decode_trans",
    }


def test_register_process_tree_ignores_processes_that_exit_during_scan():
    class ExitedProcess(FakeProcess):
        def name(self):
            raise start_utils.psutil.NoSuchProcess(self.pid)

    exited_process = ExitedProcess(pid=1001)
    router_process = FakeProcess(pid=1000, children=[exited_process])
    process_manager = start_utils.SubmoduleManager()

    process_manager.register_process_tree(router_process)

    assert process_manager.processes == []
    assert process_manager.process_names == {}


def test_setup_signal_handlers_registers_and_handles_sigterm(monkeypatch):
    http_server_process = FakeHttpServerProcess()
    process_manager = start_utils.SubmoduleManager()
    registered_handlers = {}
    terminate_calls = []
    monkeypatch.setattr(
        start_utils.signal,
        "signal",
        lambda sig, handler: registered_handlers.__setitem__(sig, handler),
    )
    monkeypatch.setattr(process_manager, "terminate_all_processes", lambda: terminate_calls.append(True))
    exit_codes = []
    monkeypatch.setattr(start_utils.os, "_exit", lambda code: exit_codes.append(code))

    process_manager.setup_signal_handlers(http_server_process)

    assert set(registered_handlers) == {
        start_utils.signal.SIGTERM,
        start_utils.signal.SIGINT,
        start_utils.signal.SIGHUP,
    }
    registered_handlers[start_utils.signal.SIGTERM](start_utils.signal.SIGTERM, None)

    assert http_server_process.sent_signals == [start_utils.signal.SIGTERM]
    assert http_server_process.wait_timeouts == [60]
    assert terminate_calls == [True]
    assert exit_codes == [0]


def test_setup_signal_handlers_uses_http_process_set_after_handler_install(monkeypatch):
    process_manager = start_utils.SubmoduleManager()
    http_server_process = FakeHttpServerProcess()
    registered_handlers = {}
    killed_processes = []
    exit_codes = []
    monkeypatch.setattr(
        start_utils.signal,
        "signal",
        lambda sig, handler: registered_handlers.__setitem__(sig, handler),
    )
    monkeypatch.setattr(start_utils, "kill_recursive", lambda process: killed_processes.append(process))
    monkeypatch.setattr(process_manager, "terminate_all_processes", lambda: None)
    monkeypatch.setattr(start_utils.os, "_exit", lambda code: exit_codes.append(code))

    process_manager.setup_signal_handlers()
    process_manager.setup_signal_handlers(http_server_process)
    registered_handlers[start_utils.signal.SIGINT](start_utils.signal.SIGINT, None)

    assert killed_processes == [http_server_process]
    assert exit_codes == [0]


def test_signal_handler_exits_even_when_cleanup_raises(monkeypatch):
    process_manager = start_utils.SubmoduleManager()
    registered_handlers = {}
    exit_codes = []
    monkeypatch.setattr(
        start_utils.signal,
        "signal",
        lambda sig, handler: registered_handlers.__setitem__(sig, handler),
    )
    monkeypatch.setattr(process_manager, "terminate_all_processes", lambda: (_ for _ in ()).throw(RuntimeError()))
    monkeypatch.setattr(start_utils.os, "_exit", lambda code: exit_codes.append(code))

    process_manager.setup_signal_handlers()
    registered_handlers[start_utils.signal.SIGINT](start_utils.signal.SIGINT, None)

    assert exit_codes == [0]


def test_terminate_skips_zombie_even_when_is_running_is_true(monkeypatch):
    zombie = FakeProcess(pid=1234, running=True)
    process_manager = start_utils.SubmoduleManager()
    process_manager.processes = [zombie]
    wait_calls = []
    monkeypatch.setattr(start_utils, "is_process_active", lambda pid: False)
    monkeypatch.setattr(start_utils.psutil, "Process", lambda pid: zombie)
    monkeypatch.setattr(start_utils.psutil, "wait_procs", lambda *args, **kwargs: wait_calls.append((args, kwargs)))

    process_manager.terminate_all_processes()

    assert zombie.kill_calls == 0
    assert wait_calls == []


def test_terminate_wait_is_bounded_and_target_pids_are_deduplicated(monkeypatch):
    child = FakeProcess(pid=1002)
    root = FakeProcess(pid=1001, children=[child])
    duplicate_root = FakeProcess(pid=1001, children=[child])
    process_by_pid = {1001: root, 1002: child}
    process_manager = start_utils.SubmoduleManager()
    process_manager.processes = [root, duplicate_root, child]
    wait_calls = []
    monkeypatch.setattr(start_utils, "is_process_active", lambda pid: True)
    monkeypatch.setattr(start_utils.psutil, "Process", lambda pid: process_by_pid[pid])
    monkeypatch.setattr(
        start_utils.psutil,
        "wait_procs",
        lambda processes, timeout: ([], wait_calls.append((processes, timeout)) or list(processes)),
    )

    process_manager.terminate_all_processes()

    assert child.kill_calls == 1
    assert root.kill_calls == 1
    assert [process.pid for process in wait_calls[0][0]] == [1002, 1001]
    assert wait_calls[0][1] == start_utils.PROCESS_SHUTDOWN_WAIT_TIMEOUT_SECONDS


def test_normal_start_installs_handlers_before_launching_submodules(monkeypatch):
    from lightllm.server import api_start

    calls = []

    class FakeManager:
        def setup_signal_handlers(self, http_server_process=None):
            calls.append(("setup", http_server_process))

        def supervise_processes(self, http_server_process):
            calls.append(("supervise", http_server_process))

    http_server_process = object()
    monkeypatch.setattr(api_start, "process_manager", FakeManager())
    monkeypatch.setattr(api_start, "_launch_subprocesses", lambda args: calls.append(("launch", None)))
    monkeypatch.setattr(
        api_start.subprocess, "Popen", lambda command: calls.append(("popen", command)) or http_server_process
    )
    monkeypatch.setattr(api_start, "get_shm_port_args", lambda: SimpleNamespace(port=8000))
    args = SimpleNamespace(
        hypercorn_config=None,
        httpserver_workers=1,
        host="127.0.0.1",
        model_dir="/model",
        health_monitor=False,
    )

    api_start.normal_or_p_d_start(args)

    assert calls[0] == ("setup", None)
    assert calls[1] == ("launch", None)
    assert calls[2][0] == "popen"
    assert calls[3] == ("setup", http_server_process)


def test_supervisor_fails_when_http_server_exits(monkeypatch):
    http_server_process = FakeHttpServerProcess(return_code=0)
    process_manager = start_utils.SubmoduleManager()
    terminate_calls = []
    kill_calls = []
    monkeypatch.setattr(process_manager, "terminate_all_processes", lambda: terminate_calls.append(True))
    monkeypatch.setattr(start_utils, "kill_recursive", lambda process: kill_calls.append(process))

    with pytest.raises(RuntimeError, match="HTTP server exited unexpectedly with return code 0"):
        process_manager.supervise_processes(http_server_process)

    assert kill_calls == []
    assert terminate_calls == [True]


def test_supervisor_fails_and_cleans_up_when_submodule_exits(monkeypatch):
    http_server_process = FakeHttpServerProcess()
    dead_process = FakeProcess(pid=1234, running=False, name="router", exitcode=-9)
    process_manager = start_utils.SubmoduleManager()
    process_manager.processes = [dead_process]
    process_manager.process_names = {dead_process: dead_process.name()}
    terminate_calls = []
    kill_calls = []
    monkeypatch.setattr(process_manager, "terminate_all_processes", lambda: terminate_calls.append(True))
    monkeypatch.setattr(start_utils, "is_process_active", lambda pid: True)
    monkeypatch.setattr(start_utils, "kill_recursive", lambda process: kill_calls.append(process))

    with pytest.raises(
        RuntimeError,
        match="Critical LightLLM submodule exited unexpectedly: name=router pid=1234 exitcode=-9",
    ):
        process_manager.supervise_processes(http_server_process)

    assert kill_calls == [http_server_process]
    assert terminate_calls == [True]


def test_supervisor_treats_zombie_submodule_as_dead(monkeypatch):
    http_server_process = FakeHttpServerProcess()
    zombie_process = FakeProcess(pid=1234, name="router", exitcode=-9)
    process_manager = start_utils.SubmoduleManager()
    process_manager.processes = [zombie_process]
    process_manager.process_names = {zombie_process: zombie_process.name()}
    terminate_calls = []
    kill_calls = []
    monkeypatch.setattr(process_manager, "terminate_all_processes", lambda: terminate_calls.append(True))
    monkeypatch.setattr(start_utils, "is_process_active", lambda pid: False)
    monkeypatch.setattr(start_utils, "kill_recursive", lambda process: kill_calls.append(process))

    with pytest.raises(
        RuntimeError,
        match="Critical LightLLM submodule exited unexpectedly: name=router pid=1234 exitcode=-9",
    ):
        process_manager.supervise_processes(http_server_process)

    assert kill_calls == [http_server_process]
    assert terminate_calls == [True]


def test_supervisor_keeps_polling_while_all_processes_are_alive(monkeypatch):
    http_server_process = FakeHttpServerProcess()
    child_process = FakeProcess(pid=1234)
    process_manager = start_utils.SubmoduleManager()
    process_manager.processes = [child_process]
    process_manager.process_names = {child_process: child_process.name()}
    terminate_calls = []
    sleep_calls = []
    monkeypatch.setattr(process_manager, "terminate_all_processes", lambda: terminate_calls.append(True))

    def stop_after_first_poll(interval):
        sleep_calls.append(interval)
        http_server_process.return_code = 2

    monkeypatch.setattr(start_utils, "is_process_active", lambda pid: True)
    monkeypatch.setattr(start_utils.time, "sleep", stop_after_first_poll)

    with pytest.raises(RuntimeError, match="HTTP server exited unexpectedly with return code 2"):
        process_manager.supervise_processes(http_server_process)

    assert sleep_calls == [5.0]
    assert terminate_calls == [True]


def test_supervisor_supports_submodule_only_processes(monkeypatch):
    dead_process = FakeProcess(pid=1234, running=False, name="model_infer", wait_timeout=True)
    process_manager = start_utils.SubmoduleManager()
    process_manager.processes = [dead_process]
    process_manager.process_names = {dead_process: dead_process.name()}
    terminate_calls = []
    kill_calls = []
    monkeypatch.setattr(process_manager, "terminate_all_processes", lambda: terminate_calls.append(True))
    monkeypatch.setattr(start_utils, "is_process_active", lambda pid: False)
    monkeypatch.setattr(start_utils, "kill_recursive", lambda process: kill_calls.append(process))

    with pytest.raises(
        RuntimeError,
        match="Critical LightLLM submodule exited unexpectedly: name=model_infer pid=1234 exitcode=None",
    ):
        process_manager.supervise_processes()

    assert kill_calls == []
    assert terminate_calls == [True]
