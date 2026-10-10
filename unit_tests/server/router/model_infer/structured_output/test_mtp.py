from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from unit_tests.server.grammar_helpers import (
    allowed,
    commit_token,
    build_mask,
    make_mask_buffers,
    apply_masks,
    make_req,
    init_request,
    prepare_sampling_tensors,
)


def verify_mask(compiler, reqs, inputs):
    mask = build_mask(reqs, draft_input_ids=inputs)
    logits = torch.zeros(len(reqs), compiler.vocab_size)
    apply_masks(reqs, logits, mask)
    return torch.isfinite(logits)


def test_verify_masks_follow_each_draft_prefix_and_roll_back(compiler):
    req = make_req(regular_constraint="abc")
    init_request(compiler, req)
    mask = verify_mask(compiler, [req] * 4, [0, ord("a"), ord("b"), ord("c")])
    assert [row.nonzero().flatten().tolist() for row in mask] == [[ord("a")], [ord("b")], [ord("c")], [256]]
    assert allowed([req])[0].nonzero().flatten().tolist() == [ord("a")]

    # Only tokens accepted by target verification become permanent history.
    commit_token(req, ord("a"))
    commit_token(req, ord("b"))
    mask = verify_mask(compiler, [req] * 2, [ord("b"), ord("c")])
    assert mask[0, ord("c")] and mask[1, 256]
    assert allowed([req])[0, ord("c")]


@pytest.mark.parametrize("draft", [b"!bc", b"a!c", b"ab!"])
def test_invalid_draft_is_rejected_without_failing_the_request(compiler, draft):
    req = make_req(regular_constraint="abc")
    init_request(compiler, req)
    mask = verify_mask(compiler, [req] * 4, [0, *draft])
    mismatch = draft.index(ord("!"))
    assert mask[mismatch, b"abc"[mismatch]]
    assert not mask[mismatch, ord("!")]
    assert mask[mismatch + 1 :].all()  # Unreachable suffix rows remain harmless.
    assert req.output_constraint.error is None
    assert allowed([req])[0, ord("a")]


def test_mixed_compacted_requests_have_independent_verify_states(compiler):
    first, second, plain = make_req(regular_constraint="ab"), make_req(regular_constraint="cd"), make_req()
    for req in (first, second, plain):
        init_request(compiler, req)
    reqs = [plain, plain, second, second, second, first, first]
    mask = verify_mask(compiler, reqs, [0, 0, 0, ord("c"), ord("d"), 0, ord("a")])
    assert mask[:2].all()
    assert [row.nonzero().flatten().tolist() for row in mask[2:]] == [
        [ord("c")],
        [ord("d")],
        [256],
        [ord("a")],
        [ord("b")],
    ]
    assert allowed([first])[0, ord("a")]
    assert allowed([second])[0, ord("c")]


@pytest.mark.parametrize("reasoning_end", [b"", b"]"])
def test_stop_token_in_draft_does_not_terminate_the_real_matcher(compiler, reasoning_end):
    req = make_req(regular_constraint="a", guided_reasoning_end=tuple(reasoning_end))
    init_request(compiler, req)
    draft = [*reasoning_end, ord("a"), 256, ord("!")]
    mask = verify_mask(compiler, [req] * (len(draft) + 1), [0, *draft])
    assert mask[: len(reasoning_end)].all()
    mask = mask[len(reasoning_end) :]
    assert mask[0, ord("a")] and mask[1, 256] and mask[2:].all()
    assert not req.output_constraint.matcher.is_terminated()
    assert req.output_constraint.error is None
    for token in reasoning_end:
        assert not commit_token(req, token)
    assert allowed([req])[0, ord("a")]


def test_reasoning_boundary_inside_verify_is_temporary_until_commit(compiler):
    req = make_req(regular_constraint="ab", guided_reasoning_end=tuple(b"]>"))
    init_request(compiler, req)
    commit_token(req, ord("]"))
    mask = verify_mask(compiler, [req] * 4, [ord("]"), ord(">"), ord("a"), ord("b")])
    assert mask[0].all()
    assert [row.nonzero().flatten().tolist() for row in mask[1:]] == [[ord("a")], [ord("b")], [256]]
    state = req.output_constraint
    assert state.in_reasoning and state.reasoning_tail == (ord("]"),)
    matcher = state.matcher
    assert matcher is not None
    # Repeating an uncommitted draft reuses the matcher at the same prefix.
    repeated_mask = verify_mask(compiler, [req] * 4, [ord("]"), ord(">"), ord("a"), ord("b")])
    assert torch.equal(repeated_mask, mask)
    assert state.matcher is matcher

    # A single MTP post_handle can commit the delimiter and the whole answer.
    for token in b">ab":
        assert not commit_token(req, token)
    assert not state.in_reasoning
    assert allowed([req])[0, 256]
    assert state.matcher is matcher
    assert commit_token(req, 256)


@pytest.mark.parametrize(
    "marker,committed,draft,expected",
    [
        (b"]>", b"", b"]>ab", [None, None, ord("a"), ord("b"), 256]),
        (b"]>", b"]", b"x]>a", [None, None, None, ord("a"), ord("b")]),
        (b"]>", b"]", b">!b", [None, ord("a"), None, None]),
        (b"]>", b"]", b"x]x", [None, None, None, None]),
        (b"aba", b"ab", b"baba", [None, None, None, None, ord("a")]),
        (b"aba", b"ab", b"aab", [None, ord("a"), ord("b"), 256]),
    ],
)
def test_verify_locates_answer_without_mutating_reasoning_progress(compiler, marker, committed, draft, expected):
    req = make_req(regular_constraint="ab", guided_reasoning_end=tuple(marker))
    init_request(compiler, req)
    for token in committed:
        commit_token(req, token)
    state = req.output_constraint
    original_tail = state.reasoning_tail

    mask = verify_mask(compiler, [req] * (len(draft) + 1), [0, *draft])

    for prediction, token in zip(mask, expected):
        if token is None:
            assert prediction.all()
        else:
            assert prediction.nonzero().flatten().tolist() == [token]
    assert state.reasoning_end_token_ids == tuple(marker)
    assert state.reasoning_tail == original_tail
    assert state.in_reasoning and state.error is None
    # The target may choose to continue thinking instead of accepting the draft.
    assert not commit_token(req, ord("?"))
    assert allowed([req])[0].all()
    # A later real transition must start from the unmodified grammar root.
    for token in marker:
        assert not commit_token(req, token)
    assert allowed([req])[0].nonzero().flatten().tolist() == [ord("a")]


@pytest.mark.parametrize("failure", ["fill_next_token_bitmask", "rollback"])
def test_mask_failure_after_speculative_reasoning_end_disables_request_mask(compiler, monkeypatch, failure):
    import xgrammar as xgr

    req = make_req(regular_constraint="ab", guided_reasoning_end=(ord("]"),))
    init_request(compiler, req)
    original = getattr(xgr.GrammarMatcher, failure)

    def fail_speculative_mask(matcher, *args):
        if failure == "rollback" or args[1] == 2:
            raise RuntimeError("injected speculative mask failure")
        return original(matcher, *args)

    with monkeypatch.context() as patcher:
        patcher.setattr(xgr.GrammarMatcher, failure, fail_speculative_mask)
        assert build_mask([req] * 5, draft_input_ids=[0, *b"]ab", 256]) is None

    state = req.output_constraint
    assert state.in_reasoning and state.reasoning_tail == ()
    assert state.error == "injected speculative mask failure"
    # An overlapping batch must not reuse partially written masks or retry the matcher.
    assert build_mask([req] * 4, draft_input_ids=[0, *b"]ab"]) is None


def test_thinking_verify_does_not_enable_mask_or_advance_matcher(compiler):
    req = make_req(regular_constraint="ab", guided_reasoning_end=(ord("]"),))
    init_request(compiler, req)
    assert build_mask([req] * 3, draft_input_ids=[0, ord("x"), ord("y")]) is None
    assert req.output_constraint.matcher.accept_token(ord("a"))


@pytest.mark.parametrize("failure", ["fill_next_token_bitmask", "rollback"])
def test_speculative_mask_failure_isolated_and_draft_tokens_rolled_back(compiler, monkeypatch, failure):
    import xgrammar as xgr

    bad = make_req(regular_constraint="ab")
    good = make_req(regular_constraint="cd")
    for req in [bad, good]:
        init_request(compiler, req)
    allowed([bad, good])
    bad_matcher = bad.output_constraint.matcher
    original = getattr(xgr.GrammarMatcher, failure)

    def fail_for_bad_request(matcher, *args):
        if matcher is bad_matcher and (failure == "rollback" or args[1] == 1):
            raise RuntimeError("injected grammar failure")
        return original(matcher, *args)

    with monkeypatch.context() as patcher:
        patcher.setattr(xgr.GrammarMatcher, failure, fail_for_bad_request)
        mask = verify_mask(compiler, [bad, bad, good], [0, ord("a"), 0])

    assert mask[:2].all()
    assert mask[2].nonzero().flatten().tolist() == [ord("c")]
    assert bad.output_constraint.error == "injected grammar failure"
    assert good.output_constraint.error is None
    if failure == "fill_next_token_bitmask":
        # The failure happened after accepting draft 'a'; cleanup must still
        # leave the matcher at its original prefix, which accepts 'a' again.
        assert bad_matcher.accept_token(ord("a"))


@pytest.mark.parametrize("dp,microbatch_side", [(False, None), (True, None), (True, 0), (True, 1)])
@pytest.mark.parametrize("dynamic", [False, True])
def test_mtp_decode_handoff_compaction_and_accepted_commits(compiler, monkeypatch, dp, microbatch_side, dynamic):
    from lightllm.server.router.model_infer.mode_backend.chunked_prefill import impl as chunked_impl
    from lightllm.server.router.model_infer.mode_backend.dp_backend import impl as dp_impl

    impl = dp_impl if dp else chunked_impl
    backend_class = dp_impl.DPChunkedPrefillBackend if dp else chunked_impl.ChunkedPrefillBackend
    backend = backend_class.__new__(backend_class)
    backend.eos_id = [256]
    backend.is_master_in_dp = False
    backend.extra_post_req_handle_func = None
    req = make_req(regular_constraint="abc")
    req.req_idx = 5
    req.out_token_id_count = {}
    init_request(compiler, req)
    order = []

    def make_input(tokens):
        return SimpleNamespace(
            batch_size=len(tokens),
            input_ids=None,
            b_req_idx=torch.full((len(tokens),), req.req_idx, dtype=torch.int32),
            b_mtp_index=torch.arange(len(tokens), dtype=torch.int32),
        )

    model_input = make_input(list(b"abcd") if dynamic else list(b"abc"))
    empty = make_input([])
    original_reqs = [req] * model_input.batch_size

    def compact(**kwargs):
        value = kwargs["model_input"]
        selection = None
        if dynamic and value.batch_size:
            value.batch_size = 3
            value.b_req_idx = value.b_req_idx[:3]
            value.b_mtp_index = value.b_mtp_index[:3]
            selection = SimpleNamespace(wait=lambda: None, tensor=torch.tensor([True, True, True, False]))
        return value, selection

    def compact_pair(**kwargs):
        first, selection0 = compact(model_input=kwargs["model_input0"])
        second, selection1 = compact(model_input=kwargs["model_input1"])
        return first, selection0, second, selection1

    def forward(value):
        assert value.input_ids is None, "Constraint preparation must not materialize model inputs"
        order.append("forward")
        logits = torch.zeros(value.batch_size, 257)
        logits[:, ord("!")] = 10  # Every constrained row must override this.
        return SimpleNamespace(logits=logits)

    def forward_pair(first, second):
        value = model_input
        output = forward(value)
        padding = SimpleNamespace(logits=torch.zeros(0, 257))
        return (output, padding) if microbatch_side == 0 else (padding, output)

    def propose(**kwargs):
        order.append("propose")
        return SimpleNamespace(token_ids=torch.empty(1, 2, dtype=torch.int64))

    engine = SimpleNamespace(
        plan_decode=lambda **kwargs: SimpleNamespace(draft_step=2, skip_verify_sync=False),
        prepare_decode_model_input=compact,
        prepare_decode_model_inputs=compact_pair,
        propose_next=propose,
        propose_next_overlap=propose,
        update_planner_statics=lambda **kwargs: None,
    )
    backend.spec_engine = backend.decode_draft_engine = engine
    backend.model = SimpleNamespace(forward=forward, microbatch_overlap_decode=forward_pair)
    monkeypatch.setattr(impl, "prepare_decode_inputs", lambda req_objs: (model_input, original_reqs))
    if microbatch_side is not None:
        layout = (
            (model_input, original_reqs, [req], empty, [], [])
            if microbatch_side == 0
            else (empty, [], [], model_input, original_reqs, [req])
        )
        monkeypatch.setattr(impl, "overlap_prepare_decode_inputs", lambda req_objs: layout)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(torch.cuda, "Event", lambda: SimpleNamespace(record=lambda: None, synchronize=lambda: None))
    monkeypatch.setattr(impl.g_infer_context, "get_overlap_stream", lambda: None)
    monkeypatch.setattr(
        impl.g_infer_context,
        "req_sampling_manager",
        SimpleNamespace(update_reqs_out_token_counter_gpu=lambda **kwargs: None),
        raising=False,
    )
    monkeypatch.setattr(
        impl.g_pin_mem_manager, "async_copy_from_gpu_tensor", lambda key, gpu_tensor: gpu_tensor.clone()
    )
    monkeypatch.setattr(impl, "gen_b_req_mtp_start_loc", lambda *args, **kwargs: torch.tensor([0]))

    token_table = torch.full((8, 4), ord("!"), dtype=torch.int64)

    class ReadyTokenTable:
        def __getitem__(self, index):
            assert "handoff" in order, "Token table read before the existing post_handle handoff"
            assert "propose" not in order, "Token table read after the next proposal could overwrite it"
            return token_table[index]

    backend._get_next_token_ranks = lambda logits, token_ids: torch.zeros_like(token_ids)
    backend._async_copy_next_token_infos_to_pin_mem = lambda next_token_ids, next_token_logprobs, next_token_ranks: (
        next_token_ids,
        next_token_logprobs,
        next_token_ranks,
    )

    def sample(logits, reqs, eos_ids, b_mtp_index):
        assert len(reqs) == 3
        assert req.out_token_id_count == {ord("a"): 1}, "MTP read stale CPU counters"
        manager = make_mask_buffers(reqs, 4)
        manager.req_to_next_token_ids = ReadyTokenTable()
        assert prepare_sampling_tensors(reqs, manager)[-1]
        assert b_mtp_index.tolist() == [0, 1, 2]
        apply_masks(reqs, logits, manager, b_mtp_index)
        order.append("sample")
        result = logits.argmax(-1)
        assert result.tolist() == [ord("b"), ord("c"), 256]
        return result, torch.zeros(3)

    monkeypatch.setattr(impl, "sample", sample)

    def verify(**kwargs):
        order.append("verify")
        assert allowed([req])[0, ord("b")], "Draft advanced permanent matcher state"
        return torch.tensor([2], dtype=torch.int32), torch.tensor([1, 1, 0], dtype=torch.int32)

    monkeypatch.setattr(impl.mtp_utils, "verify_mtp_tokens", verify)
    monkeypatch.setattr(impl.mtp_utils, "scatter_mtp_next_tokens", lambda **kwargs: None)
    backend._pre_post_handle = lambda *args, **kwargs: []

    def post(**kwargs):
        assert kwargs["next_token_ids"].tolist() == list(b"bc")
        for token in kwargs["next_token_ids"].tolist():
            commit_token(req, token)
        order.append("commit")

    backend._post_handle = post

    def handoff():
        order.append("handoff")
        token_table[req.req_idx] = torch.tensor(list(b"abcd"))
        commit_token(req, ord("a"))
        req.out_token_id_count[ord("a")] = 1

    events = SimpleNamespace(
        notify_post_handle_and_wait_pre_post_handle=handoff,
        notify_forward_and_wait_post_handle=lambda: None,
        notify_pre_post_handle=lambda: None,
    )
    method = backend.decode_overlap_mtp if microbatch_side is not None else backend.decode_mtp
    method(events, [req])
    assert order == ["forward", "handoff", "sample", "verify", "propose", "commit"]
    assert allowed([req])[0, 256]
