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

    process_manager.setup_signal_handlers(http_server_process)

    assert set(registered_handlers) == {
        start_utils.signal.SIGTERM,
        start_utils.signal.SIGINT,
        start_utils.signal.SIGHUP,
    }
    with pytest.raises(SystemExit) as exc_info:
        registered_handlers[start_utils.signal.SIGTERM](start_utils.signal.SIGTERM, None)

    assert exc_info.value.code == 0
    assert http_server_process.sent_signals == [start_utils.signal.SIGTERM]
    assert http_server_process.wait_timeouts == [60]
    assert terminate_calls == [True]


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
    dead_process = SimpleNamespace(name="router", pid=1234, exitcode=-9, is_alive=lambda: False)
    process_manager = start_utils.SubmoduleManager()
    process_manager.processes = [dead_process]
    terminate_calls = []
    kill_calls = []
    monkeypatch.setattr(process_manager, "terminate_all_processes", lambda: terminate_calls.append(True))
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
    zombie_process = SimpleNamespace(name="router", pid=1234, exitcode=None, is_alive=lambda: True)
    process_manager = start_utils.SubmoduleManager()
    process_manager.processes = [zombie_process]
    terminate_calls = []
    kill_calls = []
    monkeypatch.setattr(process_manager, "terminate_all_processes", lambda: terminate_calls.append(True))
    monkeypatch.setattr(start_utils, "is_process_active", lambda pid: False)
    monkeypatch.setattr(start_utils, "kill_recursive", lambda process: kill_calls.append(process))

    with pytest.raises(
        RuntimeError,
        match="Critical LightLLM submodule exited unexpectedly: name=router pid=1234 exitcode=None",
    ):
        process_manager.supervise_processes(http_server_process)

    assert kill_calls == [http_server_process]
    assert terminate_calls == [True]


def test_supervisor_keeps_polling_while_all_processes_are_alive(monkeypatch):
    http_server_process = FakeHttpServerProcess()
    child_process = SimpleNamespace(name="router", pid=1234, exitcode=None, is_alive=lambda: True)
    process_manager = start_utils.SubmoduleManager()
    process_manager.processes = [child_process]
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
    dead_process = SimpleNamespace(name="visual", pid=1234, exitcode=1, is_alive=lambda: False)
    process_manager = start_utils.SubmoduleManager()
    process_manager.processes = [dead_process]
    terminate_calls = []
    kill_calls = []
    monkeypatch.setattr(process_manager, "terminate_all_processes", lambda: terminate_calls.append(True))
    monkeypatch.setattr(start_utils, "kill_recursive", lambda process: kill_calls.append(process))

    with pytest.raises(
        RuntimeError,
        match="Critical LightLLM submodule exited unexpectedly: name=visual pid=1234 exitcode=1",
    ):
        process_manager.supervise_processes()

    assert kill_calls == []
    assert terminate_calls == [True]
