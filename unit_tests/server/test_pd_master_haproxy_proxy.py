"""Exercise the real streaming proxy bundled by ray-haproxy."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

import pytest
import psutil

from lightllm.server.core.objs.start_args_type import StartArgs
from lightllm.server.httpserver_for_pd_master.multiprocess import (
    _build_haproxy_config,
    _build_initial_health_state,
)
from lightllm.utils import start_utils
from lightllm.utils.start_utils import kill_recursive
from unit_tests.server.pd_master_test_utils import get_test_haproxy_binary


def test_real_haproxy_starts_with_a_low_fd_hard_limit(tmp_path):
    binary = str(Path(get_test_haproxy_binary()).resolve())
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        public_port = listener.getsockname()[1]

    args = StartArgs(
        run_mode="pd_master",
        host="127.0.0.1",
        port=public_port,
        httpserver_workers=2,
    )
    state_file = tmp_path / "haproxy-low-fd.state"
    state_file.write_text(_build_initial_health_state([1, 2]), encoding="utf-8")
    config = tmp_path / "haproxy-low-fd.cfg"
    config.write_text(_build_haproxy_config(args, [1, 2], state_file), encoding="utf-8")
    error_log_path = tmp_path / "haproxy-low-fd.stderr"
    # 同时降低 soft/hard limit，验证 HAProxy 会按实际 FD 上限自动缩小 maxconn。
    command = [
        "/bin/bash",
        "-c",
        'ulimit -n 1024 && exec "$@"',
        "lightllm-haproxy-low-fd",
        binary,
        "-db",
        "-f",
        str(config),
    ]
    with error_log_path.open("wb") as error_log:
        haproxy = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=error_log)
        try:
            deadline = time.monotonic() + 5
            while True:
                assert haproxy.poll() is None, error_log_path.read_text()
                try:
                    with socket.create_connection(("127.0.0.1", public_port), timeout=0.2):
                        break
                except OSError:
                    assert time.monotonic() < deadline, error_log_path.read_text()
                    time.sleep(0.02)
        finally:
            if haproxy.poll() is None:
                haproxy.send_signal(signal.SIGUSR1)
            haproxy.wait(timeout=5)

    assert haproxy.returncode == 0


@pytest.fixture
def haproxy_proxy(tmp_path, request):
    binary = get_test_haproxy_binary()
    state = {
        "requests": [],
        "lock": threading.Lock(),
        "release_stream": threading.Event(),
        "release_initial_probes": threading.Event(),
        "stream_disconnected": threading.Event(),
        "runtime": tmp_path,
        "probe_status": {worker: {path: 200 for path in ("/liveness", "/readiness", "/health")} for worker in range(2)},
        "probe_delays": {},
        "probe_observations": [],
    }
    if getattr(request, "param", None) != "block_initial_health_checks":
        state["release_initial_probes"].set()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path in ("/stream", "/stream_until_disconnect"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                try:
                    self.wfile.write(b"data: first\n\n")
                    self.wfile.flush()
                    if self.path == "/stream_until_disconnect":
                        for _ in range(500):
                            self.wfile.write(b"data: " + b"x" * (64 * 1024) + b"\n\n")
                            self.wfile.flush()
                            time.sleep(0.01)
                    else:
                        state["release_stream"].wait(timeout=5)
                        self.wfile.write(b"data: last\n\n")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    state["stream_disconnected"].set()
                return
            with state["lock"]:
                status = state["probe_status"][self.server.worker_id].get(self.path, 200)
                delay = state["probe_delays"].get((self.server.worker_id, self.path), 0)
            if self.path in ("/liveness", "/readiness", "/health"):
                state["release_initial_probes"].wait(timeout=15)
            if delay:
                time.sleep(delay)
            with state["lock"]:
                if self.path in ("/liveness", "/readiness", "/health"):
                    state["probe_observations"].append((self.server.worker_id, self.path, status))
            body = json.dumps({"worker_id": self.server.worker_id}).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            with state["lock"]:
                state["requests"].append((self.server.worker_id, body))
            # Simulate a worker dying after accepting a non-idempotent request.
            # A proxy retry would submit this same body to the other worker.
            self.close_connection = True
            self.connection.shutdown(socket.SHUT_RDWR)
            self.connection.close()

    servers = []
    threads = []
    haproxy = None
    try:
        for worker_id in range(2):
            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            server.daemon_threads = True
            server.worker_id = worker_id
            thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
            thread.start()
            servers.append(server)
            threads.append(thread)
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            public_port = listener.getsockname()[1]
        args = StartArgs(
            run_mode="pd_master",
            host="127.0.0.1",
            port=public_port,
            httpserver_workers=2,
        )
        worker_ports = [server.server_port for server in servers]
        state_file = tmp_path / "haproxy-health.state"
        state_file.write_text(_build_initial_health_state(worker_ports), encoding="utf-8")
        config = tmp_path / "haproxy.cfg"
        config.write_text(_build_haproxy_config(args, worker_ports, state_file), encoding="utf-8")
        executable = str(Path(binary).resolve())
        check = subprocess.run([executable, "-c", "-f", str(config)], capture_output=True, text=True, timeout=5)
        assert check.returncode == 0, check.stderr
        command = [executable, "-db", "-f", str(config)]
        with (tmp_path / "haproxy.stderr").open("wb") as error_log:
            haproxy = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=error_log, start_new_session=True)
            state["haproxy"] = haproxy
            opener = build_opener(ProxyHandler({}))
            base_url = f"http://127.0.0.1:{public_port}"
            deadline = time.monotonic() + 5
            while True:
                assert haproxy.poll() is None, (tmp_path / "haproxy.stderr").read_text()
                try:
                    with opener.open(base_url + "/worker", timeout=0.5) as response:
                        assert response.status == 200
                    break
                except (OSError, URLError):
                    if time.monotonic() >= deadline:
                        pytest.fail("HAProxy did not start: " + (tmp_path / "haproxy.stderr").read_text())
                    time.sleep(0.02)
            yield base_url, opener, state
    finally:
        state["release_stream"].set()
        state["release_initial_probes"].set()
        if haproxy is not None:
            if haproxy.poll() is None:
                haproxy.send_signal(signal.SIGUSR1)
                try:
                    haproxy.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    kill_recursive(haproxy)
            haproxy.wait(timeout=5)
            try:
                os.killpg(haproxy.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for server in servers:
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join(timeout=1)


def test_real_haproxy_balances_workers_and_delivers_sse_before_stream_completion(haproxy_proxy):
    base_url, opener, state = haproxy_proxy
    proxy_process = psutil.Process(state["haproxy"].pid)
    assert proxy_process.children(recursive=True) == []
    assert proxy_process.num_threads() >= 2
    workers = set()
    for _ in range(8):
        with opener.open(base_url + "/worker", timeout=3) as response:
            workers.add(json.load(response)["worker_id"])
    assert workers == {0, 1}

    # The backend waits for this client to read the first event before it sends
    # the last one. Buffering the complete stream would deadlock until timeout.
    with opener.open(base_url + "/stream", timeout=3) as response:
        assert response.headers["Content-Type"] == "text/event-stream"
        assert response.readline() == b"data: first\n"
        assert response.readline() == b"\n"
        assert not state["release_stream"].is_set()
        state["release_stream"].set()
        assert response.read() == b"data: last\n\n"


def test_real_haproxy_never_replays_a_post_and_keeps_pd_websockets_direct(haproxy_proxy):
    base_url, opener, state = haproxy_proxy
    body = b'{"prompt":"submit once"}'
    with pytest.raises(HTTPError) as response:
        opener.open(Request(base_url + "/generate", data=body, method="POST"), timeout=3)
    assert response.value.code == 502
    response.value.close()
    assert [request_body for _worker_id, request_body in state["requests"]] == [body]

    websocket_headers = {
        "Connection": "Upgrade",
        "Upgrade": "websocket",
        "Sec-WebSocket-Version": "13",
        "Sec-WebSocket-Key": "dGhlIHNhbXBsZSBub25jZQ==",
    }
    for path in ("/pd_register", "/kv_move_status"):
        for headers in ({}, websocket_headers):
            with pytest.raises(HTTPError) as response:
                opener.open(Request(base_url + path, headers=headers), timeout=3)
            assert response.value.code == 426
            response.value.close()


def test_real_haproxy_propagates_client_disconnect_to_the_request_worker(haproxy_proxy):
    base_url, opener, state = haproxy_proxy
    response = opener.open(base_url + "/stream_until_disconnect", timeout=3)
    assert response.readline() == b"data: first\n"
    response.close()
    assert state["stream_disconnected"].wait(timeout=3), "Backend continued producing after the client disconnected"


def test_launcher_sigterm_drains_an_active_sse_response(haproxy_proxy, monkeypatch):
    base_url, opener, state = haproxy_proxy
    registered_handlers = {}
    shutdown_result = []
    process_manager = start_utils.SubmoduleManager()
    monkeypatch.setattr(
        start_utils.signal,
        "signal",
        lambda sig, handler: registered_handlers.__setitem__(sig, handler),
    )
    monkeypatch.setattr(process_manager, "terminate_all_processes", lambda: None)
    process_manager.setup_signal_handlers(
        state["haproxy"],
        graceful_shutdown_signal=signal.SIGUSR1,
    )

    def stop_launcher():
        try:
            registered_handlers[signal.SIGTERM](signal.SIGTERM, None)
        except SystemExit as error:
            shutdown_result.append(error.code)

    with opener.open(base_url + "/stream", timeout=3) as response:
        assert response.readline() == b"data: first\n"
        assert response.readline() == b"\n"

        shutdown_thread = threading.Thread(target=stop_launcher)
        shutdown_thread.start()
        time.sleep(0.1)
        assert state["haproxy"].poll() is None, "HAProxy exited before the active stream completed"
        assert state["runtime"].is_dir()

        state["release_stream"].set()
        assert response.read() == b"data: last\n\n"
    shutdown_thread.join(timeout=5)
    assert not shutdown_thread.is_alive()
    assert shutdown_result == [0]
    assert state["haproxy"].returncode == 0


def _probe_response(opener, base_url, path, method="GET"):
    request = Request(base_url + path, data=b"" if method == "POST" else None, method=method)
    try:
        response = opener.open(request, timeout=3)
    except HTTPError as error:
        response = error
    with response:
        return response.status, response.read(), response.headers.get_content_type()


def _wait_for_probe_response(opener, base_url, path, status, timeout=15):
    deadline = time.monotonic() + timeout
    while True:
        response = _probe_response(opener, base_url, path)
        if response[0] == status:
            return response
        assert time.monotonic() < deadline, response
        time.sleep(0.05)


@pytest.mark.parametrize("haproxy_proxy", ["block_initial_health_checks"], indirect=True)
def test_real_haproxy_reports_unready_until_every_initial_probe_succeeds(haproxy_proxy):
    base_url, opener, state = haproxy_proxy

    # 后端首次探测仍在处理中时，三类健康接口及 healthz 别名必须全部返回 503。
    for path in ("/liveness", "/readiness", "/health", "/healthz"):
        assert _probe_response(opener, base_url, path)[0] == 503
    with opener.open(base_url + "/worker", timeout=3) as response:
        assert response.status == 200

    state["release_initial_probes"].set()
    for path in ("/liveness", "/readiness", "/health", "/healthz"):
        assert _wait_for_probe_response(opener, base_url, path, 200)[0] == 200


@pytest.mark.parametrize(
    "path,failure,success_body,error_body",
    [
        ("/liveness", 503, {"status": "ok"}, {"status": "not alive"}),
        ("/readiness", 302, {"status": "ok"}, {"status": "not ready"}),
        ("/health", 500, {"message": "Ok"}, {"message": "Error"}),
    ],
)
def test_real_haproxy_requires_every_worker_for_each_probe(haproxy_proxy, path, failure, success_body, error_body):
    base_url, opener, state = haproxy_proxy
    for initial_path in ("/liveness", "/readiness", "/health"):
        _wait_for_probe_response(opener, base_url, initial_path, 200)
    assert json.loads(_probe_response(opener, base_url, path)[1]) == success_body

    def change_worker_status(worker, status):
        with state["lock"]:
            before = len(state["probe_observations"])
            state["probe_status"][worker][path] = status
        # Wait for a new probe of this worker, rather than treating HAProxy's
        # initial UP state or an old response as evidence of recovery.
        deadline = time.monotonic() + 15
        while True:
            with state["lock"]:
                observed = (worker, path, status) in state["probe_observations"][before:]
            if observed:
                return
            assert time.monotonic() < deadline, (worker, path, status)
            time.sleep(0.05)

    change_worker_status(1, failure)
    status, body, content_type = _wait_for_probe_response(opener, base_url, path, 503)
    assert content_type == "application/json"
    assert json.loads(body) == error_body
    for other in ("/liveness", "/readiness", "/health"):
        if other != path:
            assert _probe_response(opener, base_url, other)[0] == 200

    # Probe failure must not filter ordinary traffic.
    for ordinary_path in ("/worker", "/metrics"):
        workers = set()
        for _ in range(6):
            with opener.open(base_url + ordinary_path, timeout=3) as response:
                workers.add(json.load(response)["worker_id"])
        assert workers == {0, 1}
    aliases = ("/health", "/healthz") if path == "/health" else (path,)
    for alias in aliases:
        assert _probe_response(opener, base_url, alias + "?detail=1")[0] == 503
    method = "HEAD" if path == "/health" else "POST"
    method_status, method_body, _ = _probe_response(opener, base_url, path, method)
    assert method_status == 503
    if method == "HEAD":
        assert method_body == b""
    else:
        assert json.loads(method_body) == error_body
        # Unsupported methods still reach the original backend handler.
        assert _probe_response(opener, base_url, path, "HEAD")[0] == 501

    change_worker_status(0, failure)
    assert _wait_for_probe_response(opener, base_url, path, 503)[0] == 503
    change_worker_status(1, 200)
    assert _wait_for_probe_response(opener, base_url, path, 503)[0] == 503
    change_worker_status(0, 200)
    assert json.loads(_wait_for_probe_response(opener, base_url, path, 200)[1]) == success_body
    if method == "HEAD":
        assert _probe_response(opener, base_url, path, method)[:2] == (200, b"")
    else:
        assert json.loads(_probe_response(opener, base_url, path, method)[1]) == success_body


def test_real_haproxy_allows_slow_health_checks_and_rejects_a_timed_out_worker(haproxy_proxy):
    base_url, opener, state = haproxy_proxy
    with state["lock"]:
        state["probe_status"][1]["/health"] = 503
    _wait_for_probe_response(opener, base_url, "/health", 503)

    # Recover from a known DOWN state so the initial UP default cannot hide a
    # check timeout that is too short for the fixed-topology eight-second probe.
    with state["lock"]:
        state["probe_status"][1]["/health"] = 200
        state["probe_delays"][(1, "/health")] = 8.5
    _wait_for_probe_response(opener, base_url, "/health", 200, timeout=25)

    with state["lock"]:
        state["probe_delays"][(1, "/health")] = 13
    _wait_for_probe_response(opener, base_url, "/health", 503, timeout=30)
    assert _probe_response(opener, base_url, "/liveness")[0] == 200
    assert _probe_response(opener, base_url, "/readiness")[0] == 200
