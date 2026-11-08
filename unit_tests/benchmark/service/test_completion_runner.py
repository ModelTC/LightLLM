from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading

import pytest
import requests

from test.benchmark.service.completion_runner import CompletionClient


@contextmanager
def service(*, overlap=False, bad_count=False):
    barrier, second_done = threading.Barrier(2), threading.Event()
    finished = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert "expected_number" not in payload
            index = payload["inputs"][0]
            if overlap:
                barrier.wait(timeout=5)
                if index == 1:
                    assert second_done.wait(timeout=5)
            body = dict(
                tokens=[dict(id=10 + index), dict(id=99)],
                prompt_tokens=1,
                count_output_tokens=1 if bad_count and index == 2 else 2,
                finish_reason="stop",
            )
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            finished.append(index)
            if index == 2:
                second_done.set()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", finished
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def cases():
    return [
        dict(dataset_index=17, prompt_ids=[1], max_output_tokens=3, expected_number="secret"),
        dict(dataset_index=3, prompt_ids=[2], max_output_tokens=3, expected_number="secret"),
    ]


def test_actual_concurrent_http_keeps_order_and_sessions_separate(tmp_path):
    sessions = []

    class Session(requests.Session):
        def __init__(self):
            super().__init__()
            self.owner, self.was_closed = threading.get_ident(), False
            sessions.append(self)

        def post(self, *args, **kwargs):
            assert self.owner == threading.get_ident() and not self.trust_env
            return super().post(*args, **kwargs)

        def close(self):
            self.was_closed = True
            super().close()

    with service(overlap=True) as (url, finished):
        client = CompletionClient(
            "lightllm",
            "model",
            url,
            10,
            [99],
            tmp_path,
            concurrency=2,
            session_factory=Session,
        )
        try:
            results = list(client.results(cases()))
        finally:
            client.close()
    assert finished == [2, 1]
    assert [x["token_ids"] for x in results] == [[11, 99], [12, 99]]
    assert len(sessions) == 2 and all(x.was_closed for x in sessions)
    assert {x.name for x in tmp_path.glob("response-*.txt")} == {"response-17.txt", "response-3.txt"}


@pytest.mark.parametrize("concurrency", [1, 2])
def test_count_failure_is_not_dropped_and_raw_response_survives(tmp_path, concurrency):
    with service(bad_count=True) as (url, finished):
        client = CompletionClient("lightllm", "model", url, 10, [99], tmp_path, concurrency=concurrency)
        try:
            results = iter(client.results(cases()))
            assert next(results)["token_ids"] == [11, 99]
            with pytest.raises(ValueError, match="counts"):
                next(results)
        finally:
            client.close()
    assert json.loads((tmp_path / "response-3.txt").read_text())["count_output_tokens"] == 1
