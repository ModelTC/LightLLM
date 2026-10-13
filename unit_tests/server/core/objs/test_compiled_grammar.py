import pickle
import asyncio
import subprocess
import sys
import uuid
from types import SimpleNamespace

import pytest
import torch

from lightllm.server.core.objs.req import Req
from lightllm.server.core.objs.sampling_params import SamplingParams
from lightllm.utils.envs_utils import get_unique_server_name
from lightllm.utils.shm_utils import create_or_link_shm


@pytest.fixture
def grammar_req(monkeypatch):
    from lightllm.server.core.objs import req as req_module

    monkeypatch.setenv("LIGHTLLM_UNIQUE_SERVICE_NAME_ID", "test_grammar_" + uuid.uuid4().hex)
    monkeypatch.setattr(
        req_module,
        "get_env_start_args",
        lambda: SimpleNamespace(mtp_step=0, model_dir="unused", enable_cpu_cache=False),
    )
    monkeypatch.setattr(req_module, "is_hybrid_att_model", lambda model_dir: False)
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


def test_compiled_payload_is_published_explicitly_after_request_initialization(compiler, grammar_req):
    sampling_params = SamplingParams()
    sampling_params.init(None, regular_constraint="ab")
    asyncio.run(sampling_params.verify_cpu_async(compiler))
    grammar_req.init(1, [0], sampling_params, None)
    with pytest.raises(FileNotFoundError):
        grammar_req.get_compiled_grammar()
    payload = asyncio.run(sampling_params.get_compiled_grammar(compiler))
    grammar_req.set_compiled_grammar(payload)
    shared_req = Req.from_buffer_copy(grammar_req)
    assert shared_req.get_compiled_grammar() == payload
    assert not hasattr(sampling_params, "compiled_grammar")
    assert not hasattr(shared_req, "compiled_grammar_size")


@pytest.mark.parametrize("previous_output_len", [-1, 2 ** 31, 2 ** 32, "1", 1.5, None])
def test_invalid_pd_previous_output_len_is_rejected_before_ctypes_conversion(previous_output_len):
    params = SamplingParams()
    with pytest.raises(ValueError, match="pd_previous_output_len"):
        params.init(None, pd_previous_output_len=previous_output_len)


@pytest.mark.parametrize("transport", ["buffer", "pickle", "dict", "origin_dict"])
@pytest.mark.parametrize(
    "constraint",
    [
        {"regular_constraint": "ab"},
        {"guided_grammar": 'root ::= "ab"'},
        {"guided_json": {"const": "ab"}},
    ],
)
def test_constraint_params_survive_transport_with_independent_storage(transport, constraint):
    params = SamplingParams()
    params.init(None, guided_reasoning_end=[7, 8], **constraint)
    params.constraint_out_params.pd_previous_output_len = 3
    original_values = params.to_dict()

    if transport == "buffer":
        copied = SamplingParams.from_buffer_copy(params)
    elif transport == "pickle":
        copied = pickle.loads(pickle.dumps(params))
    else:
        serialized = params.to_dict() if transport == "dict" else params.to_origin_dict()
        copied = SamplingParams()
        copied.init(None, **serialized)

    assert copied.constraint_out_params.guided_reasoning_end.to_list() == [7, 8]
    assert copied.constraint_out_params.pd_previous_output_len == 3
    for name in ("regular_constraint", "guided_grammar", "guided_json"):
        assert getattr(copied.constraint_out_params, name).to_str() == original_values[name]
    for name in constraint:
        getattr(copied.constraint_out_params, name).length = 0
        assert getattr(copied.constraint_out_params, name).to_str() == ""
    copied.constraint_out_params.guided_reasoning_end.sequence[0] = 9
    copied.constraint_out_params.pd_previous_output_len = 4
    assert params.constraint_out_params.guided_reasoning_end.to_list() == [7, 8]
    assert params.constraint_out_params.pd_previous_output_len == 3
    assert params.to_dict() == original_values
    assert "constraint_out_params" not in params.to_dict()


@pytest.mark.parametrize(
    "reasoning_end,history,next_output,in_reasoning",
    [
        (b"", b"", b"a", False),
        (b"", b"a", b"", False),
        (b"</think>", b"thought", b"</think>a", True),
        (b"</think>", b"thought</thi", b"nk>a", True),
        (b"</think>", b"thought</think>", b"a", False),
        (b"</think>", b"thought</think>a", b"", False),
    ],
)
@pytest.mark.parametrize("transport", ["pickle", "dict"])
def test_pd_continuation_restores_state_from_shared_prompt(
    compiler, grammar_req, reasoning_end, history, next_output, in_reasoning, transport
):
    from lightllm.server.router.model_infer.infer_batch import InferSamplingParams

    params = SamplingParams()
    params.init(
        None,
        regular_constraint="ab",
        guided_reasoning_end=list(reasoning_end),
        pd_previous_output_len=len(history),
    )
    asyncio.run(params.verify_cpu_async(compiler))
    if transport == "pickle":
        params = pickle.loads(pickle.dumps(SamplingParams.from_buffer_copy(params)))
    else:
        serialized = params.to_dict()
        params = SamplingParams()
        params.init(None, **serialized)
        asyncio.run(params.verify_cpu_async(compiler))
    # Delimiters and other text in the original prompt must never be replayed.
    prompt_ids = list(b"original </think> prompt") + list(history)
    grammar_req.init(1, prompt_ids, params, None)
    grammar_req.set_compiled_grammar(asyncio.run(params.get_compiled_grammar(compiler)))
    shared_req = Req.from_buffer_copy(grammar_req)
    shared_req.link_prompt_ids_shm_array()
    try:
        infer_params = InferSamplingParams(shared_req, vocab_size=257)
        state = compiler.grammar_cache.create_state(shared_req, infer_params)
        assert state.error is None and state.in_reasoning == in_reasoning
        assert shared_req.sample_params.constraint_out_params.pd_previous_output_len == len(history)
        for token_id in next_output:
            state.commit(token_id)
        assert state.error is None and not state.in_reasoning
        bitmask = torch.empty((1, 9), dtype=torch.int32)
        assert state.fill_masks(bitmask, [])
        allowed = [token for token in range(257) if (int(bitmask[0, token // 32]) >> (token % 32)) & 1]
        assert allowed == [ord("b")]
    finally:
        shared_req.shm_prompt_ids.detach_shm()
    params.init(None, pd_previous_output_len=99)
    assert params.constraint_out_params.pd_previous_output_len == 99
    params.init(None)
    assert params.constraint_out_params.pd_previous_output_len == 0


def test_compiled_grammar_is_readable_in_another_process(compiler, grammar_req):
    payload = asyncio.run(compiler.compile("regex", "ab"))
    grammar_req.sample_params.init(None, regular_constraint="ab")
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
    for constraint in (
        {"regular_constraint": "ab"},
        {},
        {"guided_json": {"const": {"ok": True}}},
        {},
        {"regular_constraint": "ab"},
    ):
        grammar_req.sample_params.init(None, **constraint)
        payload = asyncio.run(grammar_req.sample_params.get_compiled_grammar(compiler))
        # Unconstrained requests must ignore old slot storage even without a write.
        if payload is not None:
            grammar_req.set_compiled_grammar(payload)
        req = Req.from_buffer_copy(grammar_req)
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
