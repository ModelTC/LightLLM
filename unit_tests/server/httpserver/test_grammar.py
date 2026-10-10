import asyncio
import json
import pickle
from threading import Event
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import xgrammar as xgr

from lightllm.server.core.objs.sampling_params import SamplingParams
from unit_tests.server.grammar_helpers import configure, reset_args_cache


@pytest.fixture(autouse=True)
def start_args(monkeypatch):
    configure(monkeypatch)


def params(**kwargs):
    result = SamplingParams()
    result.init(None, **kwargs)
    return result


@pytest.fixture
def http_client(compiler, monkeypatch):
    from lightllm.server import api_http, api_openai
    from lightllm.server.api_lightllm import lightllm_generate, lightllm_generate_stream

    configure(monkeypatch, run_mode="pd_master", disable_delay_response_start=True)
    generate = MagicMock(side_effect=AssertionError("Invalid request reached generation"))
    manager = SimpleNamespace(
        tokenizer=None,
        output_grammar_compiler=compiler,
        generate=generate,
        args=SimpleNamespace(enable_return_routed_experts=False),
    )
    monkeypatch.setattr(api_http.g_objs, "httpserver_manager", manager)
    monkeypatch.setattr(api_http.g_objs, "g_generate_func", lightllm_generate)
    monkeypatch.setattr(api_http.g_objs, "g_generate_stream_func", lightllm_generate_stream)
    monkeypatch.setattr(api_http.g_objs, "metric_client", SimpleNamespace(counter_inc=lambda *args: None))
    monkeypatch.setattr(api_openai, "build_prompt", AsyncMock(return_value="test"))
    monkeypatch.setattr(api_openai, "_is_force_thinking_mode", lambda request: False)
    yield httpx.AsyncClient(transport=httpx.ASGITransport(app=api_http.app), base_url="http://test")
    generate.assert_not_called()


def assert_bad_request(client, path, body):
    async def send():
        async with client:
            return await client.post(path, json=body)

    response = asyncio.run(send())
    assert response.status_code == 400, response.text
    assert response.json()["error"]["type"] == "BadRequestError"
    return response.text


@pytest.mark.parametrize("failure", ["syntax", "exception", "timeout"])
def test_compilation_failure_returns_400_before_generation(compiler, http_client, monkeypatch, failure):
    if failure == "exception":
        monkeypatch.setattr(compiler, "_compile", MagicMock(side_effect=RuntimeError("injected compiler error")))
    elif failure == "timeout":
        compiler.timeout = 0
    grammar = "[" if failure == "syntax" else "ab"
    body = {"inputs": "test", "parameters": {"regular_constraint": grammar}}
    error = assert_bad_request(http_client, "/generate_stream", body)
    assert ("timed out" if failure == "timeout" else "Failed to compile") in error


@pytest.mark.parametrize("path", ["/generate", "/generate_stream"])
@pytest.mark.parametrize("early_headers", [False, True])
@pytest.mark.parametrize("invalid_params", [{"max_new_tokens": 0}, {"do_sample": True, "top_p": 0}])
def test_invalid_sampling_params_return_400(http_client, monkeypatch, path, early_headers, invalid_params):
    configure(monkeypatch, disable_delay_response_start=early_headers)
    assert_bad_request(http_client, path, {"inputs": "test", "parameters": invalid_params})


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/completions"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("failure", ["syntax", "exception", "timeout"])
def test_openai_grammar_failures_return_400(compiler, http_client, monkeypatch, path, stream, failure):
    if failure == "exception":
        monkeypatch.setattr(compiler, "_compile", MagicMock(side_effect=RuntimeError("injected compiler error")))
    elif failure == "timeout":
        compiler.timeout = 0
    schema = {"type": 42} if failure == "syntax" else {"type": "object"}
    body = {
        "model": "test",
        "stream": stream,
        "response_format": {"type": "json_schema", "json_schema": {"name": "test", "schema": schema}},
    }
    body.update({"messages": []} if "chat" in path else {"prompt": "test"})
    assert_bad_request(http_client, path, body)


@pytest.mark.parametrize(
    "constraint,text",
    [
        ({"guided_grammar": 'root ::= "ab"'}, b"ab"),
        ({"regular_constraint": "ab"}, b"ab"),
        ({"guided_json": {"const": "ok"}}, b'"ok"'),
    ],
)
def test_artifact_survives_pd_copy_and_pickle_without_recompilation(compiler, monkeypatch, constraint, text):
    sampling_params = params(**constraint)
    asyncio.run(sampling_params.verify_async(compiler))
    # Choices and PD continuation each copy the scalar struct. Both that copy
    # and the existing network pickle must preserve the serialized artifact.
    copied = pickle.loads(pickle.dumps(sampling_params.copy().copy()))
    assert copied.compiled_grammar == sampling_params.compiled_grammar
    with monkeypatch.context() as patcher:
        patcher.setattr(xgr, "GrammarCompiler", MagicMock(side_effect=AssertionError("Backend recompiled")))
        grammar = compiler.grammar_cache.get_grammar(copied.compiled_grammar)
        matcher = xgr.GrammarMatcher(grammar)
        assert all(matcher.accept_token(token) for token in text)
        assert matcher.accept_token(256) and matcher.is_terminated()
    assert "compiled_grammar" not in sampling_params.to_dict()
    copied.max_new_tokens = 2
    assert sampling_params.max_new_tokens != copied.max_new_tokens


def test_completed_artifact_is_returned_without_executor_work(compiler, monkeypatch):
    async def run():
        payload = await compiler.compile("regex", "ab")
        monkeypatch.setattr(
            compiler._executor, "submit", MagicMock(side_effect=AssertionError("Cache hit reached executor"))
        )
        assert await compiler.compile("regex", "ab") is payload

    asyncio.run(run())


def test_completed_artifact_cache_evicts_least_recently_used(compiler, monkeypatch):
    compiler._cache_size = 2
    compile_grammar = MagicMock(wraps=compiler._compile)
    monkeypatch.setattr(compiler, "_compile", compile_grammar)

    async def run():
        for grammar in ("a", "b", "a", "c", "a", "b"):
            await compiler.compile("regex", grammar)

    asyncio.run(run())
    assert [call.args[1] for call in compile_grammar.call_args_list] == ["a", "b", "c", "b"]


def test_failed_compilation_is_retried_before_caching_success(compiler, monkeypatch):
    compile_grammar = MagicMock(side_effect=[RuntimeError("temporary failure"), b"compiled artifact"])
    monkeypatch.setattr(compiler, "_compile", compile_grammar)

    async def run():
        with pytest.raises(ValueError, match="temporary failure"):
            await compiler.compile("regex", "ab")
        payload = await compiler.compile("regex", "ab")
        assert payload == b"compiled artifact"
        assert await compiler.compile("regex", "ab") is payload

    asyncio.run(run())
    assert compile_grammar.call_count == 2


def test_no_grammar_does_not_use_compiler_and_none_mode_rejects_constraints():
    compiler = MagicMock(side_effect=AssertionError("Ordinary request compiled"))
    asyncio.run(params().verify_async(compiler))
    compiler.compile.assert_not_called()
    with pytest.raises(ValueError, match="output_constraint_mode xgrammar"):
        asyncio.run(params(regular_constraint="ab").verify_async(None))


@pytest.mark.parametrize("stop_waiter", ["cancel", "timeout"])
def test_cold_requests_compile_independently_and_cache_only_success(compiler, monkeypatch, stop_waiter):
    release = Event()
    compile_grammar = compiler._compile

    async def run():
        loop = asyncio.get_running_loop()
        started, finished = asyncio.Event(), asyncio.Event()

        def slow_first_compile(*args):
            loop.call_soon_threadsafe(started.set)
            try:
                assert release.wait(5)
                return b"late artifact"
            finally:
                loop.call_soon_threadsafe(finished.set)

        monkeypatch.setattr(compiler, "_compile", slow_first_compile)
        compiler.timeout = 0.02 if stop_waiter == "timeout" else 5
        first = asyncio.create_task(compiler.compile("regex", "ab"))
        try:
            await asyncio.wait_for(started.wait(), timeout=5)
            # The same grammar can finish on another worker while the first job
            # is blocked. Neither a cache hit nor shared pending work is involved.
            second_compile = MagicMock(wraps=compile_grammar)
            monkeypatch.setattr(compiler, "_compile", second_compile)
            compiler.timeout = 5
            payload = await compiler.compile("regex", "ab")
            second_compile.assert_called_once_with("regex", "ab")

            if stop_waiter == "cancel":
                first.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await first
            else:
                with pytest.raises(ValueError, match="timed out"):
                    await first
        finally:
            release.set()

        await asyncio.wait_for(finished.wait(), timeout=5)
        # The abandoned worker's eventual result must not replace a cached success.
        assert await compiler.compile("regex", "ab") is payload

    try:
        asyncio.run(run())
    finally:
        release.set()


@pytest.mark.parametrize("explicit_eos", [None, [255]])
def test_pd_master_compiler_resolves_eos_before_model_loading(monkeypatch, tmp_path, explicit_eos):
    from lightllm.server.core.objs import StartArgs
    from lightllm.server.httpserver.grammar import create_output_grammar_compiler

    (tmp_path / "config.json").write_text('{"model_type":"llama","vocab_size":257,"eos_token_id":256}')
    args = StartArgs(run_mode="pd_master", model_dir=str(tmp_path), eos_id=explicit_eos)
    stop_ids = explicit_eos if explicit_eos is not None else [256]
    info = xgr.TokenizerInfo([bytes([i]) for i in range(256)] + [b"<eos>"], stop_token_ids=stop_ids)
    tokenizer = object()
    with patch.object(xgr.TokenizerInfo, "from_huggingface", return_value=info) as create_info:
        compiler = create_output_grammar_compiler(args, tokenizer)
    try:
        create_info.assert_called_once_with(tokenizer, vocab_size=257, stop_token_ids=stop_ids)
        payload = asyncio.run(compiler.compile("regex", "a"))
        matcher = xgr.GrammarMatcher(xgr.CompiledGrammar.deserialize_json(payload.decode(), info))
        assert matcher.accept_token(ord("a"))
        assert matcher.accept_token(stop_ids[0]) and matcher.is_terminated()
    finally:
        compiler.shutdown()


def test_dictionary_schema_is_serialized_during_initialization():
    params = SamplingParams()
    params.init(None, guided_json={"type": "object"})
    assert json.loads(params.guided_json.to_str()) == {"type": "object"}


@pytest.mark.parametrize(
    "invalid_params", [{"max_new_tokens": 0}, {"do_sample": True, "top_p": 0}, {"best_of": 2, "n": 1}]
)
def test_parameter_validation_precedes_grammar_compilation(invalid_params):
    params = SamplingParams()
    params.init(None, regular_constraint="ab", **invalid_params)
    compiler = SimpleNamespace(compile=AsyncMock())
    with pytest.raises(ValueError):
        asyncio.run(params.verify_async(compiler))
    compiler.compile.assert_not_called()
    assert not params.compiled_grammar


@pytest.mark.parametrize(
    "constraint", [{"guided_grammar": "bad grammar"}, {"guided_json": "not json"}, {"regular_constraint": "["}]
)
def test_invalid_grammar_is_rejected_by_async_validation(compiler, constraint):
    params = SamplingParams()
    params.init(None, **constraint)
    with pytest.raises(ValueError):
        asyncio.run(params.verify_async(compiler))
    assert not params.compiled_grammar


@pytest.mark.parametrize("stop_match", [False, True])
def test_http_preserves_error_status_and_inflight_token(stop_match):
    import numpy as np

    from lightllm.server.core.objs import FinishStatus, Req
    from lightllm.server.httpserver.manager import HttpServerManager, ReqStatus
    from lightllm.server.pd_io_struct import NodeRole

    req = Req()
    req.request_id = req.group_req_id = 1
    req.input_len = 1
    req.shm_prompt_ids = SimpleNamespace(arr=np.array([ord("P"), ord("!")]))
    req.shm_logprobs = SimpleNamespace(arr=np.array([(0.0,), (-0.2,)], dtype=[("logprob", np.float32)]))
    req.finish_status.set_status(FinishStatus.FINISHED_ERROR)
    req.finish_token_index = req.stop_str_matched_token_index = 1
    req.stop_str_matched = stop_match
    req.out_tokens_queue.push("!", 1, False, 1)
    req.get_output_logprobs_metadata = MagicMock(return_value={"token": "!"})
    req.merge_final_token_metadata = AsyncMock()

    async def collect():
        manager = HttpServerManager.__new__(HttpServerManager)
        manager.args = SimpleNamespace(use_reward_model=False, enable_return_routed_experts=False)
        manager.tokenizer = None
        manager.is_multinode_tp_slave = False
        manager.pd_mode = NodeRole.NORMAL
        wakeups = asyncio.Queue()
        wakeups.put_nowait(None)
        manager.zmq_recv_socket = SimpleNamespace(recv_pyobj=wakeups.get)
        manager.recycle_resource_loop = AsyncMock()
        status = ReqStatus(1, None, [req], 0)
        manager.req_id_to_out_inf = {1: status}
        task = asyncio.create_task(manager.handle_loop())
        try:
            await asyncio.wait_for(status.event.wait(), timeout=2)
            return status.out_token_info_list[0]
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    _, text, metadata, finish = asyncio.run(collect())
    assert text == "!" and finish.is_finished_error()
    assert metadata["id"] == ord("!") and metadata["logprob"] == pytest.approx(-0.2)
    assert metadata["logprobs"] == {"token": "!"}
    req.merge_final_token_metadata.assert_awaited_once()
