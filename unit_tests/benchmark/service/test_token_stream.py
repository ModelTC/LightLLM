import json

import pytest

from test.benchmark.service.token_stream import SSEDecoder, TokenStream


def test_sse_handles_split_utf8_crlf_and_multiline_json():
    decoder = SSEDecoder()
    raw = ': heartbeat\r\ndata: {"text": "中",\r\ndata: "token_ids": [7, 8]}\r\n\r\ndata: [DONE]\n\n'.encode()
    events = []
    for byte in raw:
        events.extend(decoder.feed(bytes([byte])))
    decoder.finish()
    assert json.loads(events[0]) == {"text": "中", "token_ids": [7, 8]}
    assert events[1] == "[DONE]"


@pytest.mark.parametrize("raw", [b'data: {"x": 1}', b'data: {"x": 1}\n', b'data: "\xe4'])
def test_truncated_sse_cannot_pass(raw):
    decoder = SSEDecoder()
    decoder.feed(raw)
    with pytest.raises((ValueError, UnicodeDecodeError)):
        decoder.finish()


@pytest.mark.parametrize("framework", ["vllm", "sglang"])
def test_burst_token_count_and_arrival_metrics(framework):
    stream = TokenStream(framework, [11, 12], 3, 1.0)
    stream.record(json.dumps({"choices": [{"index": 0, "text": "ab", "token_ids": [7, 8]}]}), 1.5)
    stream.record(
        json.dumps({"choices": [{"index": 0, "text": "c", "token_ids": [9], "finish_reason": "length"}]}), 1.8
    )
    stream.record(json.dumps({"choices": [], "usage": {"prompt_tokens": 2, "completion_tokens": 3}}), 1.9)
    stream.record("[DONE]", 1.9)
    result = stream.summarize(2.0)
    assert result["output_tokens"] == 3 and result["events"] == 3
    assert result["ttft_s"] == 0.5
    assert result["tpot_s"] == pytest.approx(0.15)
    assert result["receiver_itl_s"] == pytest.approx([0, 0.3])


def test_native_empty_text_is_still_an_output_token():
    stream = TokenStream("lightllm", [11], 1, 1.0)
    stream.record(
        json.dumps(
            {
                "token": {"id": 7, "text": "", "prompt_tokens": 1, "count_output_tokens": 1},
                "finished": True,
                "finish_reason": "length",
            }
        ),
        1.2,
    )
    assert stream.summarize(1.3)["output_tokens"] == 1


@pytest.mark.parametrize("framework", ["lightllm", "vllm", "sglang"])
@pytest.mark.parametrize("cached", [None, 0, 1, 2])
def test_optional_cache_counts_preserve_unknown_and_zero(framework, cached):
    stream = TokenStream(framework, [11, 12], 1, 1.0)
    if framework == "lightllm":
        body = dict(token=dict(id=7, prompt_tokens=2, count_output_tokens=1), finished=True, finish_reason="length")
        if cached is not None:
            body["token"]["prompt_cache_len"] = cached
    else:
        body = dict(
            choices=[dict(index=0, token_ids=[7], finish_reason="length")],
            usage=dict(prompt_tokens=2, completion_tokens=1),
        )
        if cached is not None:
            body["usage"]["prompt_tokens_details"] = dict(cached_tokens=cached)
    stream.record(json.dumps(body), 1.2)
    if framework != "lightllm":
        stream.record("[DONE]", 1.3)
    result = stream.summarize(1.4)
    assert result["cached_input_tokens"] == cached
    assert result["uncached_input_tokens"] == (None if cached is None else 2 - cached)


@pytest.mark.parametrize("cached", [-1, True, 3, 1.0])
def test_invalid_cached_usage_cannot_pass(cached):
    stream = TokenStream("vllm", [11, 12], 1, 1.0)
    with pytest.raises(ValueError, match="Cached input"):
        stream.record(
            json.dumps(
                dict(
                    choices=[],
                    usage=dict(prompt_tokens=2, completion_tokens=0, prompt_tokens_details=dict(cached_tokens=cached)),
                )
            ),
            1.1,
        )


def test_cached_usage_cannot_change_within_request():
    stream = TokenStream("vllm", [11, 12], 1, 1.0)
    stream.record(
        json.dumps(
            dict(
                choices=[],
                usage=dict(prompt_tokens=2, completion_tokens=0, prompt_tokens_details=dict(cached_tokens=1)),
            )
        ),
        1.1,
    )
    with pytest.raises(ValueError, match="changed"):
        stream.record(
            json.dumps(
                dict(
                    choices=[],
                    usage=dict(prompt_tokens=2, completion_tokens=0, prompt_tokens_details=dict(cached_tokens=0)),
                )
            ),
            1.2,
        )


@pytest.mark.parametrize("cold", [False, True])
def test_native_transport_requests_observable_details_and_explicit_cold_policy(monkeypatch, tmp_path, cold):
    import requests
    from test.benchmark.service.token_stream import request_stream

    class Response:
        status_code = 200
        headers = {"content-type": "text/event-stream", "transfer-encoding": "chunked"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def iter_content(self, chunk_size):
            body = dict(
                token=dict(id=7, prompt_tokens=2, count_output_tokens=1, prompt_cache_len=0 if cold else 1),
                finished=True,
                finish_reason="length",
            )
            yield ("data:" + json.dumps(body) + "\n\n").encode()

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, url, json, **kwargs):
            assert json["parameters"]["return_details"] is True
            assert json["parameters"].get("disable_prompt_cache", False) is cold
            return Response()

    monkeypatch.setattr(requests, "Session", Session)
    result = request_stream(
        "http://localhost", "lightllm", "model", [11, 12], 1, 10, tmp_path / "response.sse", disable_prompt_cache=cold
    )
    assert result["metrics"]["cached_input_tokens"] == (0 if cold else 1)


def test_baseline_request_cannot_silently_ignore_native_cache_policy(tmp_path):
    from test.benchmark.service.token_stream import request_stream

    with pytest.raises(ValueError, match="only implemented for native"):
        request_stream(
            "http://localhost", "vllm", "model", [11], 1, 10, tmp_path / "response.sse", disable_prompt_cache=True
        )


@pytest.mark.parametrize(
    "body",
    [
        {"choices": [{"index": 0, "text": "x"}]},
        {"choices": [{"index": 0, "token_ids": [True]}]},
        {"choices": [{"index": 0, "token_ids": [7, 8, 9]}]},
        {"choices": [{"index": 0, "token_ids": [7], "finish_reason": "stop"}]},
        {"choices": [{"index": 0, "prompt_token_ids": [99]}]},
        {"choices": [], "usage": {"prompt_tokens": 99, "completion_tokens": 0}},
        {"error": {"message": "aborted"}},
        {"choices": [{"index": 0, "token_ids": [-1]}]},
        {"choices": [{"index": 0, "token_ids": 0}]},
    ],
)
def test_invalid_stream_events_are_rejected(body):
    stream = TokenStream("vllm", [11], 2, 1.0)
    with pytest.raises(ValueError):
        stream.record(json.dumps(body), 1.1)


def test_truncated_generation_and_missing_done_cannot_be_reported():
    stream = TokenStream("vllm", [11], 1, 1.0)
    with pytest.raises(ValueError):
        stream.record("[DONE]", 1.1)
    with pytest.raises(ValueError):
        stream.summarize(1.2)
    stream.record(json.dumps({"choices": [{"index": 0, "token_ids": [7], "finish_reason": "length"}]}), 1.2)
    with pytest.raises(ValueError):
        stream.summarize(1.3)
    with pytest.raises(ValueError):
        stream.record(json.dumps({"choices": [{"index": 0, "token_ids": [8]}]}), 1.3)


def test_control_events_also_require_monotonic_timestamps():
    stream = TokenStream("vllm", [11], 1, 1.0)
    stream.record(json.dumps({"choices": []}), 1.5)
    with pytest.raises(ValueError):
        stream.record(json.dumps({"choices": []}), 1.4)
    stream.record(json.dumps({"choices": [{"index": 0, "token_ids": [7], "finish_reason": "length"}]}), 1.6)
    stream.record("[DONE]", 1.7)
    with pytest.raises(ValueError):
        stream.summarize(1.65)


@pytest.mark.parametrize("framework", ["vllm", "sglang"])
@pytest.mark.parametrize("cache_salt", [None, "isolated-prefix", "x" * 128])
def test_transport_uses_chunk_boundaries_and_saves_raw_after_timing(monkeypatch, tmp_path, framework, cache_salt):
    import requests
    import time
    from test.benchmark.service.token_stream import request_stream

    bodies = [
        {"choices": [{"index": 0, "text": "ab", "token_ids": [7, 8]}]},
        {"choices": [{"index": 0, "token_ids": [9], "finish_reason": "length"}]},
        {"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 3}},
    ]
    chunks = [("data: " + json.dumps(body) + "\n\n").encode() for body in bodies]
    chunks.append(b"data: [DONE]\n\n")

    class Response:
        status_code = 200
        headers = {"content-type": "text/event-stream", "transfer-encoding": "chunked"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def iter_content(self, chunk_size):
            assert chunk_size is None
            yield from chunks

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, url, json, **kwargs):
            assert json["prompt"] == [11] and json["return_token_ids"] is True
            assert json["stream_options"]["include_usage"] is True
            if cache_salt is None:
                assert "cache_salt" not in json
            else:
                assert json["cache_salt"] == cache_salt
            return Response()

    monkeypatch.setattr(requests, "Session", Session)
    timestamps = iter([1.0, 1.5, 1.8, 1.9, 1.9, 2.0])
    monkeypatch.setattr(time, "perf_counter", lambda: next(timestamps))
    path = tmp_path / "response.sse"
    result = request_stream("http://localhost", framework, "model", [11], 3, 10, path, cache_salt=cache_salt)
    assert result["token_ids"] == [7, 8, 9]
    assert result["metrics"]["ttft_s"] == 0.5
    assert result["metrics"]["e2e_s"] == 1.0
    assert result["timeline"] == {"started": 1.0, "first_token": 1.5, "last_token": 1.8, "ended": 2.0}
    assert path.read_bytes() == b"".join(chunks)


@pytest.mark.parametrize("cache_salt", ["", "x" * 129, 1, True, ["salt"], "x@y", "x/y", "x\\y", "x\x00y"])
def test_invalid_cache_salt_is_rejected_before_transport(tmp_path, cache_salt):
    from test.benchmark.service.token_stream import request_stream

    with pytest.raises(ValueError, match="Cache salt must"):
        request_stream(
            "http://localhost", "vllm", "model", [11], 1, 10, tmp_path / "response.sse", cache_salt=cache_salt
        )


def test_native_cache_salt_cannot_be_silently_ignored(tmp_path):
    from test.benchmark.service.token_stream import request_stream

    with pytest.raises(ValueError, match="only implemented for baseline"):
        request_stream(
            "http://localhost", "lightllm", "model", [11], 1, 10, tmp_path / "response.sse", cache_salt="salt"
        )
