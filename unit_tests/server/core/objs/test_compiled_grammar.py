import pickle
import asyncio
import subprocess
import sys
import uuid

import pytest

from lightllm.server.core.objs.req import Req
from lightllm.server.core.objs.sampling_params import SamplingParams
from lightllm.utils.envs_utils import get_unique_server_name
from lightllm.utils.shm_utils import create_or_link_shm


@pytest.fixture
def grammar_req(monkeypatch):
    monkeypatch.setenv("LIGHTLLM_UNIQUE_SERVICE_NAME_ID", "test_grammar_" + uuid.uuid4().hex)
    get_unique_server_name.cache_clear()
    req = Req()
    req.index_in_shm_mem = 3
    yield req
    try:
        shm = create_or_link_shm("shm_grammar_3", -1, force_mode="link")
        shm.close()
        shm.unlink()
    except FileNotFoundError:
        pass
    for attr in ("shm_prompt_ids", "shm_logprobs"):
        if hasattr(req, attr):
            getattr(req, attr).close_shm()
    get_unique_server_name.cache_clear()


def test_request_initialization_publishes_compiled_payload(compiler, grammar_req, monkeypatch):
    from types import SimpleNamespace
    from lightllm.server.core.objs import req as req_module

    monkeypatch.setattr(
        req_module,
        "get_env_start_args",
        lambda: SimpleNamespace(mtp_step=0, model_dir="unused", enable_cpu_cache=False),
    )
    monkeypatch.setattr(req_module, "is_hybrid_att_model", lambda model_dir: False)
    sampling_params = SamplingParams()
    sampling_params.init(None, regular_constraint="ab")
    asyncio.run(sampling_params.verify_async(compiler))
    grammar_req.init(1, [0], sampling_params, None)
    shared_req = Req.from_buffer_copy(grammar_req)
    assert shared_req.get_compiled_grammar() == sampling_params.compiled_grammar
    # The fixed-size ctypes sampling struct deliberately does not carry bytes.
    assert shared_req.sample_params.compiled_grammar == b""


def test_compiled_grammar_is_readable_in_another_process(compiler, grammar_req):
    payload = asyncio.run(compiler.compile("regex", "ab"))
    grammar_req.set_compiled_grammar(payload)
    # Only the fixed request struct crosses the process boundary. The compiled
    # artifact itself must be loaded from the HTTP-written shared memory.
    req = Req.from_buffer_copy(grammar_req)
    child = """
import pickle, sys
import xgrammar as xgr
req = pickle.loads(sys.stdin.buffer.read())
info = xgr.TokenizerInfo([bytes([i]) for i in range(256)] + [b"<eos>"], stop_token_ids=[256])
grammar = xgr.CompiledGrammar.deserialize_json(req.get_compiled_grammar().decode(), info)
matcher = xgr.GrammarMatcher(grammar)
assert not matcher.accept_token(ord("!"))
assert matcher.accept_token(ord("a")) and matcher.accept_token(ord("b"))
assert matcher.accept_token(256) and matcher.is_terminated()
"""
    result = subprocess.run([sys.executable, "-c", child], input=pickle.dumps(req), capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr.decode()


def test_reused_request_slot_does_not_expose_previous_grammar(compiler, grammar_req):
    first = asyncio.run(compiler.compile("regex", "ab"))
    second = asyncio.run(compiler.compile("json", '{"const":{"ok":true}}'))
    for payload in (first, b"", second, b"", first):
        grammar_req.set_compiled_grammar(payload)
        req = Req.from_buffer_copy(grammar_req)
        assert req.compiled_grammar_size == len(payload)
        assert req.get_compiled_grammar() == payload


def test_backend_rejects_incompatible_tokenizer_metadata(compiler):
    import xgrammar as xgr
    from lightllm.server.router.model_infer.structured_output.grammar_cache import OutputGrammarCache
    from unittest.mock import patch

    payload = asyncio.run(compiler.compile("regex", "ab"))
    other_info = xgr.TokenizerInfo([bytes([i]) for i in range(256)] + [b"<eos>"], stop_token_ids=[255])
    with patch.object(xgr.TokenizerInfo, "from_huggingface", return_value=other_info):
        cache = OutputGrammarCache(object(), 257, [255])
    with pytest.raises(Exception, match="[Mm]etadata"):
        cache.get_grammar(payload)
