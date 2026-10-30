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


def test_start_submodule_processes_returns_and_manages_psutil_processes(monkeypatch):
    class FakePipeReader:
        def recv(self):
            # 子进程应在等待初始化结果之前就进入 manager，保证此时 Ctrl-C 可以清理它们。
            assert len(process_manager.processes) == 2
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


def test_setup_signal_handlers_uses_configured_graceful_shutdown_signal(monkeypatch):
    http_server_process = FakeHttpServerProcess()
    process_manager = start_utils.SubmoduleManager()
    registered_handlers = {}
    monkeypatch.setattr(
        start_utils.signal,
        "signal",
        lambda sig, handler: registered_handlers.__setitem__(sig, handler),
    )
    monkeypatch.setattr(process_manager, "terminate_all_processes", lambda: None)

    process_manager.setup_signal_handlers(
        http_server_process,
        graceful_shutdown_signal=start_utils.signal.SIGUSR1,
    )

    with pytest.raises(SystemExit) as exc_info:
        registered_handlers[start_utils.signal.SIGTERM](start_utils.signal.SIGTERM, None)

    assert exc_info.value.code == 0
    assert http_server_process.sent_signals == [start_utils.signal.SIGUSR1]
    assert http_server_process.wait_timeouts == [60]


def test_setup_exit_controller_starts_cleanup_process_without_registering_signals(monkeypatch):
    process_manager = start_utils.SubmoduleManager()
    cleanup_process_calls = []
    registered_handlers = {}
    monkeypatch.setattr(start_utils, "get_unique_server_name", lambda: "service_0")
    monkeypatch.setattr(
        start_utils,
        "start_launcher_shm_cleanup_process",
        lambda service_name: cleanup_process_calls.append(service_name),
    )
    monkeypatch.setattr(
        start_utils.signal,
        "signal",
        lambda sig, handler: registered_handlers.__setitem__(sig, handler),
    )

    process_manager.setup_exit_controller()

    assert cleanup_process_calls == ["service_0"]
    assert registered_handlers == {}


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


def test_readiness_waits_for_bound_endpoint_and_pd_topology(monkeypatch):
    from unittest.mock import Mock

    logger = Mock()
    monkeypatch.setattr(start_utils, "logger", logger)
    monkeypatch.setattr(start_utils.time, "monotonic", lambda: 25)
    flag = Mock()
    args = SimpleNamespace(run_mode="pd_master", model_name="test-model")
    state = start_utils.ServiceReadiness(flag, args, "http://127.0.0.1:8000", 10)
    state.observe(None)
    logger.warning.assert_not_called()
    assert not state.probe_failure_logged
    state.observe(500)
    logger.warning.assert_called_once()
    assert logger.warning.call_args.kwargs["extra"]["http_status"] == 500
    logger.warning.reset_mock()
    state.observe(503)
    state.observe(503)
    assert logger.info.call_count == 1
    flag.set_value.assert_not_called()
    state.observe(200)
    state.observe(200)
    assert logger.info.call_count == 2
    flag.set_value.assert_called_once_with(1)
    assert logger.info.call_args.kwargs["extra"]["event"] == "service.ready"
    assert logger.info.call_args.kwargs["extra"]["startup_s"] == 15
    state.observe(None)
    state.observe(None)
    logger.warning.assert_not_called()
    state.observe(None)
    logger.warning.assert_not_called()
    assert state.ready
    state.observe(503)
    state.observe(503)
    state.observe(503)
    logger.warning.assert_called_once()
    assert logger.warning.call_args.kwargs["extra"]["reason"] == "readiness_probe_failed"
    assert flag.set_value.call_args.args == (0,)
    state.observe(200)
    assert not state.ready
    state.observe(200)
    assert logger.info.call_args.kwargs["extra"]["event"] == "service.recovered"
    assert logger.info.call_args.kwargs["extra"]["uptime_s"] == 15


@pytest.mark.parametrize(
    "mode,host,expected",
    [
        ("normal", "0.0.0.0", "http://127.0.0.1:8000/readiness"),
        ("pd_master", "::", "http://[::1]:8000/readiness"),
        ("config_server", "0.0.0.0", "http://127.0.0.2:9000/readiness"),
    ],
)
def test_readiness_monitor_probes_bound_address(monkeypatch, mode, host, expected):
    import requests
    from unittest.mock import Mock, MagicMock
    from lightllm.server.router.dynamic_prompt import shared_arr
    from lightllm.utils import shm_port_args

    manager = start_utils.SubmoduleManager()
    flag = Mock()
    shared_int = Mock(return_value=flag)
    monkeypatch.setattr(shared_arr, "SharedInt", shared_int)
    monkeypatch.setattr(shm_port_args, "get_shm_port_args", lambda: SimpleNamespace(port=8000, config_server_port=9000))
    monkeypatch.setattr(start_utils.threading, "Thread", lambda target, **kw: SimpleNamespace(start=target))
    session = MagicMock()
    session.__enter__.return_value = session

    def probe(url, timeout):
        assert url == expected
        assert timeout == 1
        assert session.trust_env is False
        return SimpleNamespace(status_code=200)

    monkeypatch.setattr(manager.readiness_stop, "wait", lambda timeout: manager.readiness_stop.set())
    session.get.side_effect = probe
    monkeypatch.setattr(requests, "Session", lambda: session)
    args = SimpleNamespace(run_mode=mode, host=host, config_server_host="127.0.0.2", model_name="test")
    manager.start_readiness_monitor(FakeHttpServerProcess(), args)
    session.get.assert_called_once()
    shared_int.assert_called_once_with("service_ready")
    assert [call.args for call in flag.set_value.call_args_list] == [(0,), (1,)]


def test_readiness_monitor_leaves_missing_config_host_to_supervisor(monkeypatch):
    from unittest.mock import Mock
    from lightllm.server.router.dynamic_prompt import shared_arr

    monkeypatch.setattr(shared_arr, "SharedInt", Mock())
    thread = Mock()
    monkeypatch.setattr(start_utils.threading, "Thread", thread)
    args = SimpleNamespace(run_mode="config_server", config_server_host=None)
    start_utils.SubmoduleManager().start_readiness_monitor(FakeHttpServerProcess(), args)
    thread.assert_not_called()


@pytest.mark.parametrize("failure", [None, 503, 500])
def test_readiness_tolerates_isolated_failures_and_confirms_recovery(monkeypatch, failure):
    from unittest.mock import Mock

    logger = Mock()
    monkeypatch.setattr(start_utils, "logger", logger)
    flag = Mock()
    state = start_utils.ServiceReadiness(flag, SimpleNamespace(run_mode="normal", model_name="test"), "http://test", 0)
    state.observe(200)
    for _ in range(2):
        state.observe(failure)
        assert state.ready
    state.observe(200)
    logger.warning.assert_not_called()
    flag.set_value.assert_called_once_with(1)
    for _ in range(3):
        state.observe(failure)
    if failure is None:
        assert state.ready
        flag.set_value.assert_called_once_with(1)
        logger.warning.assert_not_called()
        return
    assert not state.ready
    assert flag.set_value.call_args.args == (0,)
    logger.warning.assert_called_once()
    state.observe(200)
    assert not state.ready
    state.observe(failure)
    state.observe(200)
    assert not state.ready
    state.observe(200)
    assert state.ready
    assert logger.info.call_args.kwargs["extra"]["event"] == "service.recovered"


@pytest.mark.parametrize("failure,reason", [("timeout", "probe_timeout"), ("connection", "probe_connection_failed")])
def test_readiness_monitor_transport_failures_preserve_ready_and_stop_after_http_exit(monkeypatch, failure, reason):
    import requests
    from unittest.mock import Mock, MagicMock
    from lightllm.server.router.dynamic_prompt import shared_arr
    from lightllm.utils import shm_port_args

    manager = start_utils.SubmoduleManager()
    logger = Mock()
    flag = Mock()
    monkeypatch.setattr(start_utils, "logger", logger)
    monkeypatch.setattr(shared_arr, "SharedInt", lambda _: flag)
    monkeypatch.setattr(shm_port_args, "get_shm_port_args", lambda: SimpleNamespace(port=8000))
    monkeypatch.setattr(start_utils.threading, "Thread", lambda target, **kw: SimpleNamespace(start=target))
    session = MagicMock()
    session.__enter__.return_value = session
    error = requests.Timeout if failure == "timeout" else requests.ConnectionError
    session.get.side_effect = [SimpleNamespace(status_code=200), error(), error(), error()]
    monkeypatch.setattr(requests, "Session", lambda: session)
    process = Mock()
    process.poll.side_effect = [None] * 8 + [1]
    monkeypatch.setattr(manager.readiness_stop, "wait", lambda _: None)
    args = SimpleNamespace(run_mode="normal", host="127.0.0.1", model_name="test")
    manager.start_readiness_monitor(process, args)
    assert session.get.call_count == 4
    logger.warning.assert_not_called()
    assert [call.args for call in flag.set_value.call_args_list] == [(0,), (1,)]
    session.__exit__.assert_called_once()


@pytest.mark.parametrize(
    "poll_results,probe_count,ready", [([0], 0, False), ([None, 0], 1, False), ([None, None, 0], 1, True)]
)
def test_readiness_monitor_stops_when_http_process_exits(monkeypatch, poll_results, probe_count, ready):
    import requests
    from unittest.mock import Mock, MagicMock
    from lightllm.server.router.dynamic_prompt import shared_arr
    from lightllm.utils import shm_port_args

    manager = start_utils.SubmoduleManager()
    flag = Mock()
    monkeypatch.setattr(shared_arr, "SharedInt", lambda name: flag)
    monkeypatch.setattr(shm_port_args, "get_shm_port_args", lambda: SimpleNamespace(port=8000))
    monkeypatch.setattr(start_utils.threading, "Thread", lambda target, **kw: SimpleNamespace(start=target))
    session = MagicMock()
    session.__enter__.return_value = session
    session.get.return_value = SimpleNamespace(status_code=200)
    monkeypatch.setattr(requests, "Session", lambda: session)
    monkeypatch.setattr(manager.readiness_stop, "wait", Mock())
    process = Mock()
    process.poll.side_effect = poll_results
    args = SimpleNamespace(run_mode="normal", host="127.0.0.1", model_name="test")
    manager.start_readiness_monitor(process, args)
    assert session.get.call_count == probe_count
    assert [call.args for call in flag.set_value.call_args_list] == ([(0,), (1,)] if ready else [(0,)])
    session.__exit__.assert_called_once()


def test_readiness_monitor_tls(monkeypatch, tmp_path):
    import ssl
    import subprocess
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from unittest.mock import Mock
    from lightllm.server.router.dynamic_prompt import shared_arr
    from lightllm.utils import shm_port_args

    cert, key = tmp_path / "cert.pem", tmp_path / "key.pem"
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-days",
            "1",
            "-subj",
            "/CN=service.example",
            "-keyout",
            str(key),
            "-out",
            str(cert),
        ],
        check=True,
        capture_output=True,
    )
    config = tmp_path / "hypercorn.toml"
    config.write_text(f'certfile = "{cert}"\nkeyfile = "{key}"\n')

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            assert self.path == "/readiness"
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    manager = start_utils.SubmoduleManager()
    flag = Mock()
    monkeypatch.setattr(shared_arr, "SharedInt", lambda name: flag)
    monkeypatch.setattr(shm_port_args, "get_shm_port_args", lambda: SimpleNamespace(port=server.server_port))
    monkeypatch.setattr(start_utils.threading, "Thread", lambda target, **kw: SimpleNamespace(start=target))
    monkeypatch.setattr(manager.readiness_stop, "wait", lambda timeout: manager.readiness_stop.set())
    args = SimpleNamespace(run_mode="normal", host="0.0.0.0", model_name="test", hypercorn_config=str(config))
    try:
        manager.start_readiness_monitor(FakeHttpServerProcess(), args)
        assert [call.args for call in flag.set_value.call_args_list] == [(0,), (1,)]
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


@pytest.mark.parametrize("identity", [None, "separate", "combined"])
def test_readiness_monitor_mutual_tls(monkeypatch, tmp_path, identity):
    import ssl
    import subprocess
    import requests
    from unittest.mock import MagicMock, Mock
    from lightllm.server.router.dynamic_prompt import shared_arr
    from lightllm.utils import shm_port_args

    for name in ("server", "client"):
        subprocess.run(
            [
                "openssl",
                "req",
                "-x509",
                "-newkey",
                "rsa:2048",
                "-nodes",
                "-days",
                "1",
                "-subj",
                f"/CN={name}",
                "-keyout",
                str(tmp_path / f"{name}.key"),
                "-out",
                str(tmp_path / f"{name}.pem"),
            ],
            check=True,
            capture_output=True,
        )
    config = tmp_path / "hypercorn.toml"
    config.write_text(
        f'certfile = "{tmp_path / "server.pem"}"\nkeyfile = "{tmp_path / "server.key"}"\n'
        f'ca_certs = "{tmp_path / "client.pem"}"\nverify_mode = 2\n'
    )
    monkeypatch.delenv("LIGHTLLM_READINESS_CLIENT_CERT", raising=False)
    monkeypatch.delenv("LIGHTLLM_READINESS_CLIENT_KEY", raising=False)
    if identity == "separate":
        monkeypatch.setenv("LIGHTLLM_READINESS_CLIENT_CERT", str(tmp_path / "client.pem"))
        monkeypatch.setenv("LIGHTLLM_READINESS_CLIENT_KEY", str(tmp_path / "client.key"))
    elif identity == "combined":
        combined = tmp_path / "identity.pem"
        combined.write_bytes((tmp_path / "client.pem").read_bytes() + (tmp_path / "client.key").read_bytes())
        monkeypatch.setenv("LIGHTLLM_READINESS_CLIENT_CERT", str(combined))

    from hypercorn.__main__ import _load_config

    server_context = _load_config(str(config)).create_ssl_context()
    assert server_context.verify_mode == ssl.CERT_REQUIRED
    session = MagicMock()
    session.__enter__.return_value = session

    def probe(url, timeout):
        assert url == "https://127.0.0.1:8000/readiness"
        context = session.mount.call_args.args[1].poolmanager.connection_pool_kw["ssl_context"]
        server_in, server_out, client_in, client_out = [ssl.MemoryBIO() for _ in range(4)]
        server = server_context.wrap_bio(server_in, server_out, server_side=True)
        client = context.wrap_bio(client_in, client_out, server_side=False, server_hostname="server")
        finished = set()
        try:
            for _ in range(20):
                for peer, outgoing, incoming in ((client, client_out, server_in), (server, server_out, client_in)):
                    try:
                        peer.do_handshake()
                        finished.add(peer)
                    except ssl.SSLWantReadError:
                        pass
                    incoming.write(outgoing.read())
                if len(finished) == 2:
                    assert server.getpeercert()
                    return SimpleNamespace(status_code=200)
        except ssl.SSLError as error:
            assert identity is None
            assert error.reason == "PEER_DID_NOT_RETURN_A_CERTIFICATE"
            raise requests.exceptions.SSLError(str(error)) from error
        pytest.fail("TLS handshake did not finish")

    manager = start_utils.SubmoduleManager()
    flag = Mock()
    logger = Mock()
    monkeypatch.setattr(start_utils, "logger", logger)
    monkeypatch.setattr(shared_arr, "SharedInt", lambda name: flag)
    monkeypatch.setattr(shm_port_args, "get_shm_port_args", lambda: SimpleNamespace(port=8000))
    monkeypatch.setattr(requests, "Session", lambda: session)
    monkeypatch.setattr(start_utils.threading, "Thread", lambda target, **kw: SimpleNamespace(start=target))
    monkeypatch.setattr(manager.readiness_stop, "wait", lambda timeout: manager.readiness_stop.set())
    session.get.side_effect = probe
    args = SimpleNamespace(run_mode="normal", host="0.0.0.0", model_name="test", hypercorn_config=str(config))
    manager.start_readiness_monitor(FakeHttpServerProcess(), args)
    session.get.assert_called_once()
    assert [call.args for call in flag.set_value.call_args_list] == ([(0,)] if identity is None else [(0,), (1,)])
    if identity is None:
        logger.warning.assert_called_once()
        assert logger.warning.call_args.kwargs["extra"]["reason"] == "probe_tls_failed"
    else:
        logger.warning.assert_not_called()
