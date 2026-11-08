import importlib
import json
from pathlib import Path
import threading

import pytest
import requests


@pytest.fixture
def client_module(monkeypatch):
    root = Path(__file__).resolve().parents[3]
    monkeypatch.syspath_prepend(str(root / "test/benchmark/qwen38"))
    return importlib.import_module("check_scheduler")


def response_for(monkeypatch, module, body, http_error=False):
    class Response:
        text = json.dumps(body)

        def raise_for_status(self):
            if http_error:
                raise requests.HTTPError("503")

        def json(self):
            return body

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, url, json, timeout):
            assert json["parameters"]["do_sample"] is False
            assert json["parameters"]["return_details"] is True
            assert json["parameters"]["print_eos_token"] is True
            return Response()

    monkeypatch.setattr(module.requests, "Session", Session)


@pytest.mark.parametrize(
    "ids,ignore_eos,reason", [([3, 4], True, "length"), ([248046], False, "stop"), ([3, 248044], False, "stop")]
)
def test_valid_termination_preserves_raw_response(client_module, monkeypatch, tmp_path, ids, ignore_eos, reason):
    body = {
        "tokens": [{"id": token} for token in ids],
        "prompt_tokens": 3,
        "count_output_tokens": len(ids),
        "finish_reason": reason,
    }
    response_for(monkeypatch, client_module, body)
    case = {"inputs": [7, 8, 9], "max_new_tokens": 2, "ignore_eos": ignore_eos}
    path = tmp_path / "response.txt"
    result = client_module.generate("http://localhost", case, [248044, 248046], 10, path)
    assert result["token_ids"] == ids and json.loads(path.read_text()) == body


@pytest.mark.parametrize(
    "ids,ignore_eos,reason,prompt,count",
    [
        ([], True, "length", 3, 0),
        ([True, 4], True, "length", 3, 2),
        ([-1, 4], True, "length", 3, 2),
        ([3, 4, 5], True, "length", 3, 3),
        ([3], True, "length", 3, 1),
        ([3, 4], True, "stop", 3, 2),
        ([3, 4], True, "length", 2, 2),
        ([3, 4], True, "length", 3, 1),
        ([3], False, "length", 3, 1),
        ([248046, 3], False, "stop", 3, 2),
        ([3, 248046], False, "length", 3, 2),
    ],
)
def test_invalid_response_cannot_pass(client_module, monkeypatch, tmp_path, ids, ignore_eos, reason, prompt, count):
    body = {
        "tokens": [{"id": token} for token in ids],
        "prompt_tokens": prompt,
        "count_output_tokens": count,
        "finish_reason": reason,
    }
    response_for(monkeypatch, client_module, body)
    case = {"inputs": [7, 8, 9], "max_new_tokens": 2, "ignore_eos": ignore_eos}
    path = tmp_path / "response.txt"
    with pytest.raises(ValueError):
        client_module.generate("http://localhost", case, [248044, 248046], 10, path)
    assert json.loads(path.read_text()) == body


def test_http_error_is_saved(client_module, monkeypatch, tmp_path):
    body = {"error": "service unavailable"}
    response_for(monkeypatch, client_module, body, http_error=True)
    path = tmp_path / "response.txt"
    with pytest.raises(requests.HTTPError):
        client_module.generate("http://localhost", {"inputs": [7]}, [248046], 10, path)
    assert json.loads(path.read_text()) == body


def test_monitor_records_server_batch_gauges(client_module, monkeypatch, tmp_path):
    stop = threading.Event()
    payload = '# a comment\nlightllm_num_running_reqs{model_name="test"} 4.0\nlightllm_batch_current_size 3.0\n'

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get(self, *args, **kwargs):
            stop.set()
            return type("Response", (), {"text": payload, "raise_for_status": lambda self: None})()

    monkeypatch.setattr(client_module.requests, "Session", Session)
    summary = {"maximum_router_batch": 0, "samples": 0}
    path = tmp_path / "metrics.txt"
    client_module.monitor("http://localhost", stop, summary, path)
    assert summary == {"maximum_router_batch": 4.0, "samples": 1}
    assert path.read_text() == payload
