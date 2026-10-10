import asyncio
from itertools import count
from types import MethodType, SimpleNamespace
from weakref import ref
from unittest.mock import patch

import pytest
import torch
import xgrammar as xgr

from lightllm.server.core.objs import FinishStatus
from lightllm.server.httpserver.grammar import OutputGrammarCompiler
from unit_tests.server.grammar_helpers import (
    make_req,
    init_request,
    make_mask_buffers,
    prepare_sampling_tensors,
    build_mask,
    apply_masks,
    allowed,
    commit_token,
)
from lightllm.server.router.model_infer import infer_batch


def test_sampling_preparation_without_constraint_buffers():
    req = make_req()
    manager = SimpleNamespace(req_to_bitmask=None, req_to_bitmask_enabled=None)
    tensors = prepare_sampling_tensors([req, req], manager)
    assert tensors[0].tolist() == [req.req_idx, req.req_idx]
    assert tensors[1].tolist() == [1.0, 1.0]
    assert not tensors[-1]


@pytest.mark.parametrize("mode", ["none", "xgrammar"])
def test_context_initializes_artifact_cache_with_resolved_eos_at_startup(monkeypatch, mode):
    args = SimpleNamespace(
        output_constraint_mode=mode, model_dir="unused", tokenizer_mode="auto", trust_remote_code=False, eos_id=[255]
    )
    monkeypatch.setattr(infer_batch, "get_env_start_args", lambda: args)
    tokenizer = SimpleNamespace(eos_token_id=256)
    tokenizer_info = xgr.TokenizerInfo([bytes([i]) for i in range(256)] + [b"<eos>"], stop_token_ids=[255])
    payload = xgr.GrammarCompiler(tokenizer_info).compile_regex("a").serialize_json().encode()
    context = infer_batch.InferenceContext()
    with (
        patch("lightllm.server.tokenizer.get_tokenizer", return_value=tokenizer) as load_tokenizer,
        patch.object(xgr.TokenizerInfo, "from_huggingface", return_value=tokenizer_info) as build_tokenizer_info,
        patch.object(xgr, "GrammarCompiler", side_effect=AssertionError("Inference must not compile")),
    ):
        context.register(
            backend=SimpleNamespace(dp_rank_in_node=0),
            req_manager=SimpleNamespace(req_sampling_params_manager=None),
            radix_cache=None,
            shm_req_manager=None,
            vocab_size=257,
        )
        if mode == "none":
            assert context.output_grammar_cache is None
            load_tokenizer.assert_not_called()
            build_tokenizer_info.assert_not_called()
            return
        load_tokenizer.assert_called_once_with("unused", "auto", trust_remote_code=False)
        build_tokenizer_info.assert_called_once_with(tokenizer, vocab_size=257, stop_token_ids=[255])
        req = make_req(regular_constraint="a")
        req.shm_req.get_compiled_grammar = lambda: payload
        req.output_constraint = context.output_grammar_cache.create_state(req.shm_req, req.sampling_param)
        commit_token(req, ord("a"))
        assert req.output_constraint.matcher.accept_token(255)
        assert req.output_constraint.is_terminated()


def test_thinking_does_not_advance_matcher_or_fill_mask(compiler):
    req = make_req(regular_constraint="ab", guided_reasoning_end=(ord("]"),))
    init_request(compiler, req)
    manager = make_mask_buffers([req])
    manager.req_to_bitmask.fill_(42)
    assert not prepare_sampling_tensors([req], manager)[-1]
    assert (manager.req_to_bitmask == 42).all()
    assert not commit_token(req, ord("?"))
    assert req.output_constraint.error is None
    assert not commit_token(req, ord("]"))
    mask = allowed([req])
    assert mask[0, ord("a")] and not mask[0, ord("?")]


@pytest.mark.parametrize("marker", [b"]", b"aba"])
def test_reasoning_boundary_mixed_batch_and_shared_schema(compiler, marker):
    thinking = make_req(guided_json='{"const":{"ok":true}}', guided_reasoning_end=tuple(marker))
    direct = make_req(guided_json='{"const":{"ok":true}}')
    for req in (thinking, direct):
        init_request(compiler, req)
    assert compiler.grammar_cache.get_grammar.cache_info().currsize == 1
    mask = allowed([thinking, direct])
    assert mask[0].all() and not mask[1, ord("?")]
    assert thinking.output_constraint.matcher is not direct.output_constraint.matcher

    # A failed partial marker must not activate constraints. The prompt is ignored.
    for token in b"thought aab?" + marker[:-1]:
        assert not commit_token(thinking, token)
        assert allowed([thinking, direct])[0].all()
    # Sampling preparation observes the delimiter committed by post_handle.
    assert not commit_token(thinking, marker[-1])
    mask = build_mask([thinking, direct])
    logits = torch.zeros(2, compiler.vocab_size)
    apply_masks([thinking, direct], logits, mask)
    assert torch.isfinite(logits[:, ord("{")]).all()
    assert not torch.isfinite(logits[:, ord("?")]).any()

    for token in b'{"ok":true}':
        assert allowed([thinking])[0, token]
        assert not commit_token(thinking, token)
    assert allowed([thinking])[0, 256]
    assert commit_token(thinking, 256)
    assert allowed([direct])[0, ord("{")]


def test_unconstrained_batch_does_not_compile_or_allocate_mask(compiler):
    req = make_req()
    init_request(compiler, req)
    assert allowed([req]).all()
    assert req.output_constraint is None
    assert not compiler._cache


def test_request_admission_loads_shared_artifacts_without_compilation(compiler, monkeypatch):
    payload = asyncio.run(compiler.compile("regex", "ab"))
    reqs = {
        1: make_req(regular_constraint="ab"),
        2: make_req(regular_constraint="ab"),
        3: make_req(),
        4: make_req(regular_constraint="ab"),
    }
    for index in (1, 2, 4):
        reqs[index].shm_req.get_compiled_grammar = lambda: payload
    monkeypatch.setattr(compiler, "_compile", lambda *args: pytest.fail("Inference must not compile"))
    monkeypatch.setattr(infer_batch, "InferReq", lambda req_id, **kwargs: reqs[req_id])
    monkeypatch.setattr(infer_batch, "get_env_start_args", lambda: SimpleNamespace(diverse_mode=False))
    context = infer_batch.InferenceContext(
        req_manager=SimpleNamespace(alloc=count().__next__),
        requests_mapping={},
        output_grammar_cache=compiler.grammar_cache,
    )
    context.backend = SimpleNamespace(dp_rank_in_node=0)
    context.infer_req_ids = []
    context.vocab_size = compiler.vocab_size
    # Another DP rank's request is a temporary KV donor and needs no matcher.
    context.add_reqs([(1, 0, None, 0), (2, 1, None, 0), (3, 2, None, 0), (4, 3, None, 1)])
    assert reqs[3].output_constraint is None and reqs[4].output_constraint is None
    assert compiler.grammar_cache.get_grammar.cache_info().misses == 1
    assert reqs[1].output_constraint.matcher is not reqs[2].output_constraint.matcher
    mask = allowed([reqs[1], reqs[2], reqs[3]])
    assert mask[:2, ord("a")].all() and mask[2].all()
    commit_token(reqs[1], ord("a"))
    mask = allowed([reqs[2], reqs[1]])
    assert mask[0, ord("a")] and not mask[0, ord("b")]
    assert mask[1, ord("b")] and not mask[1, ord("a")]


@pytest.mark.parametrize("failure", ["read", "deserialize", "matcher"])
def test_artifact_load_failure_finishes_only_its_request(compiler, monkeypatch, failure):
    bad = make_req(regular_constraint="ab")
    good = make_req(regular_constraint="cd")
    init_request(compiler, good)
    payload = asyncio.run(compiler.compile("regex", "ab"))

    def read_artifact():
        if failure == "read":
            raise FileNotFoundError("missing compiled artifact")
        return b"invalid artifact" if failure == "deserialize" else payload

    def fail_to_create_matcher(*args):
        raise RuntimeError("matcher failed")

    bad.shm_req.get_compiled_grammar = read_artifact
    with monkeypatch.context() as patcher:
        if failure == "matcher":
            patcher.setattr(xgr, "GrammarMatcher", fail_to_create_matcher)
        bad.output_constraint = compiler.grammar_cache.create_state(bad.shm_req, bad.sampling_param)
    assert bad.output_constraint.error is not None
    assert build_mask([bad]) is None
    bad.set_next_gen_token_id = lambda *args, **kwargs: None
    bad.update_finish_status = lambda **kwargs: None
    infer_batch.InferReqUpdatePack(bad, 1).handle(ord("!"), 0.0, -1, [256], is_master_in_dp=False)
    assert bad.finish_status.is_finished_error()
    assert allowed([good])[0].nonzero().flatten().tolist() == [ord("c")]


def test_constraint_state_lifetime_follows_request_cleanup(compiler):
    req = make_req(regular_constraint="ab")
    req.req_idx = 7
    init_request(compiler, req)
    allowed([req])
    state_ref = ref(req.output_constraint)
    payload = req.shm_req.get_compiled_grammar()
    grammar = compiler.grammar_cache.get_grammar(payload)
    context = infer_batch.InferenceContext(
        req_manager=SimpleNamespace(free=lambda *args: None),
        requests_mapping={1: req},
        shm_req_manager=SimpleNamespace(put_back_req_obj=lambda req: None),
    )
    context.infer_req_ids = [1]
    context.args = SimpleNamespace(diverse_mode=False)
    context.backend = SimpleNamespace(is_master_in_dp=False)
    context.free_a_req_mem = lambda *args: None

    context._filter([1], modify_shm_finish_state=False)
    assert context.requests_mapping == {} and context.infer_req_ids == []
    del req
    assert state_ref() is None
    assert compiler.grammar_cache.get_grammar(payload) is grammar


def test_mixed_batch_partial_prefill_and_finished_requests(compiler):
    partial = make_req(regular_constraint="ab")
    active = make_req(regular_constraint="cd")
    plain = make_req()
    for req in [partial, active, plain]:
        init_request(compiler, req)
    mask = allowed([partial, active, plain], [False, True, True])
    assert mask[0].all() and mask[2].all()
    assert mask[1, ord("c")] and not mask[1, ord("a")]
    assert allowed([partial])[0].nonzero().flatten().tolist() == [ord("a")]

    active.finish_status.set_status(FinishStatus.FINISHED_STOP)
    assert build_mask([active]) is None


@pytest.mark.parametrize("is_master_in_dp", [False, True])
@pytest.mark.parametrize(
    "reasoning_end,first_token,expected_next",
    [(b"", ord("a"), ord("b")), (b"]", ord("]"), ord("a")), (b"]>", ord("]"), None), (b"", ord("!"), None)],
)
def test_pd_first_token_updates_constraint_before_sampling(
    compiler, monkeypatch, is_master_in_dp, reasoning_end, first_token, expected_next
):
    from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend

    req = make_req(regular_constraint="ab", guided_reasoning_end=tuple(reasoning_end))
    good = make_req(regular_constraint="cd")
    for request in (req, good):
        init_request(compiler, request)
    req.pd_task_success_num = 0
    req.shm_req.shm_prompt_ids.arr.append(0)
    req.shm_req.shm_logprobs = SimpleNamespace(arr=[None, None])
    req.set_next_gen_token_id = MethodType(infer_batch.InferReq.set_next_gen_token_id, req)
    req.update_finish_status = lambda **kwargs: None
    sampling_params = make_mask_buffers([req, good])
    transfer = SimpleNamespace(
        request_id=req.req_idx,
        has_error=False,
        first_gen_token_id=first_token,
        first_gen_token_logprob=0.0,
    )
    backend = SimpleNamespace(
        shm_pd_trans_io_buffer=SimpleNamespace(read_obj=lambda: [transfer], sub_state=lambda: None),
        is_pd_decode_mode=True,
        model=SimpleNamespace(req_manager=SimpleNamespace(req_sampling_params_manager=sampling_params)),
        eos_id=[256],
        is_master_in_dp=is_master_in_dp,
    )
    monkeypatch.setattr(infer_batch.g_infer_context, "requests_mapping", {req.req_idx: req})
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: SimpleNamespace(synchronize=lambda: None))

    # Exercise the real PD receive handler and InferReqUpdatePack.handle()
    # before any sampling preparation can update the constraint.
    ModeBackend._read_pd_trans_io_buffer_and_update_req_status(backend)

    assert req.cur_output_len == 1 and req.pd_task_success_num == 1
    assert req.shm_req.shm_prompt_ids.arr == [ord("P"), first_token]
    assert sampling_params.req_to_next_token_ids[req.req_idx, 0].item() == first_token
    if is_master_in_dp:
        assert req.shm_req.shm_cur_output_len == 1
    state = req.output_constraint
    if first_token == ord("!"):
        assert req.finish_status.status == FinishStatus.FINISHED_ERROR
        assert "Grammar rejected sampled token" in state.error
        assert allowed([good])[0].nonzero().flatten().tolist() == [ord("c")]
        assert good.output_constraint.error is None
        return

    assert state.error is None and not req.finish_status.is_finished()
    if expected_next is None:
        assert state.in_reasoning and state.reasoning_tail == (first_token,)
        assert allowed([req])[0].all()
        assert not commit_token(req, ord(">"))
        expected_next = ord("a")

    # Repeated mask preparation must observe the same committed prefix.
    for _ in range(2):
        mask = allowed([req, good])
        assert [row.nonzero().flatten().tolist() for row in mask] == [[expected_next], [ord("c")]]


@pytest.mark.parametrize(
    "constraints,text",
    [
        ({"guided_grammar": 'root ::= "hello"'}, "hello"),
        (
            {
                "guided_json": '{"type":"object","properties":{"x":{"const":1}},'
                '"required":["x"],"additionalProperties":false}'
            },
            '{"x":1}',
        ),
        ({"guided_grammar": "json"}, '{"x":1}'),
    ],
)
def test_grammar_and_schema_enforce_each_token(compiler, constraints, text):
    req = make_req(**constraints)
    init_request(compiler, req)
    for token in text.encode():
        mask = allowed([req])
        assert mask[0, token]
        req.output_constraint.commit(token)
        assert not req.output_constraint.is_terminated()
    assert allowed([req])[0, 256]
    req.output_constraint.commit(256)
    assert req.output_constraint.is_terminated()


def test_json_object_does_not_accept_array_root(compiler):
    req = make_req(guided_grammar="json")
    init_request(compiler, req)
    mask = allowed([req])
    assert mask[0, ord("{")]
    assert not mask[0, ord("[")]


def test_wrapped_tokenizer_uses_underlying_hf_tokenizer(monkeypatch):
    tokenizer = object()
    wrapper = SimpleNamespace(tokenizer=SimpleNamespace(tokenizer=tokenizer))
    received = []

    def from_huggingface(value, **kwargs):
        received.append(value)
        return xgr.TokenizerInfo([bytes([i]) for i in range(256)] + [b"<eos>"], stop_token_ids=[256])

    monkeypatch.setattr(xgr.TokenizerInfo, "from_huggingface", from_huggingface)
    compiler = OutputGrammarCompiler(wrapper, 257, [256])
    asyncio.run(compiler.compile("regex", "ab"))
    compiler.shutdown()
    assert received == [tokenizer]
