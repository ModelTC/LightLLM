from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from unit_tests.server.grammar_helpers import (
    allowed,
    commit_token,
    make_req,
    init_request,
)
from lightllm.common.req_manager import req_sampling_params
from lightllm.common.basemodel.triton_kernel.mtp_utils import mtp_verify
from lightllm.server.router.model_infer.mode_backend import generic_post_process as sampling
from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend
from lightllm.server.router.model_infer.mtp_speculative.utils import scatter_mtp_next_tokens


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Sampling integration requires CUDA")


@pytest.fixture
def sampling_manager(request, monkeypatch):
    args = SimpleNamespace(
        penalty_counter_mode=getattr(request, "param", "cpu_counter"),
        model_dir="unused",
        mtp_step=3,
        mtp_dynamic_verify=True,
        output_constraint_mode="xgrammar",
    )
    monkeypatch.setattr(req_sampling_params, "get_env_start_args", lambda: args)
    monkeypatch.setattr(req_sampling_params, "get_vocab_size", lambda model_dir: 257)
    result = req_sampling_params.ReqSamplingParamsManager(8)
    assert result.req_to_next_token_ids.is_cuda
    assert result.req_to_next_token_ids_cpu.device.type == "cpu" and result.req_to_next_token_ids_cpu.is_pinned()
    assert result.req_to_next_token_scores.is_cuda
    monkeypatch.setattr(sampling.g_infer_context, "req_manager", SimpleNamespace(req_sampling_params_manager=result))
    yield result
    torch.cuda.synchronize()


def sampling_req(compiler, sampling_manager, req_idx=5, **constraints):
    req = make_req(**({"regular_constraint": "abc"} | constraints))
    req.req_idx = req_idx
    req.sampling_param.shm_param = SimpleNamespace(
        constraint_out_params=req.sampling_param.shm_param.constraint_out_params,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        repetition_penalty=1.0,
        exponential_decay_length_penalty=SimpleNamespace(to_tuple=lambda: (0, 1.1)),
        input_penalty=False,
        min_new_tokens=0,
        temperature=1.0,
        top_p=1.0,
        top_k=1,
        output_top_logprobs=-1,
    )
    req.get_last_gen_token = lambda: req.shm_req.shm_prompt_ids.arr[-1]
    init_request(compiler, req)
    sampling_manager.init_req_sampling_params(req)
    return req


def test_sample_disables_stale_constraint_mask_for_ordinary_requests(compiler, sampling_manager, monkeypatch):
    req = sampling_req(compiler, sampling_manager, regular_constraint=None)
    sampling_manager.req_to_bitmask_enabled[req.req_idx] = True
    sampling_manager.req_to_bitmask[req.req_idx].zero_()

    def unexpected_mask(*args, **kwargs):
        pytest.fail("Ordinary requests must not launch the constraint mask kernel")

    monkeypatch.setattr(sampling, "apply_constraint_mask", unexpected_mask)
    logits = torch.zeros(1, 257, device="cuda")
    logits[:, ord("!")] = 10
    next_tokens, logprobs, _ = sampling.sample(logits, [req], [256])

    assert next_tokens.tolist() == [ord("!")]
    assert torch.isfinite(logprobs).all()
    assert not sampling_manager.req_to_bitmask_enabled[req.req_idx]
    assert not compiler._cache


@pytest.mark.parametrize("topk", [0, 1, 3, 20])
@pytest.mark.parametrize("sampling_case", ["greedy", "random", "top_k_top_p"])
def test_sample_returns_requested_output_logprobs(compiler, sampling_manager, monkeypatch, topk, sampling_case):
    req = sampling_req(compiler, sampling_manager, regular_constraint=None)
    req.sampling_param.shm_param.output_top_logprobs = topk
    req.sampling_param.shm_param.top_k = 1 if sampling_case == "greedy" else req.vocab_size
    if sampling_case == "top_k_top_p":
        req.sampling_param.shm_param.top_k = 50
        req.sampling_param.shm_param.top_p = 0.8
        monkeypatch.setattr(sampling, "get_env_start_args", lambda: SimpleNamespace(sampling_backend="triton"))
    logits = torch.linspace(-2, 2, req.vocab_size, device="cuda").view(1, -1)
    token_ids, logprobs, output_topk_logprobs = sampling.sample(logits, [req], [256])
    reference = logits.log_softmax(dim=-1)
    sampled_id = int(token_ids[0])
    expected = reference[0, sampled_id]
    if int((logits[0] > logits[0, sampled_id]).sum()) >= 20:
        expected = expected.new_tensor(-9999.0)
    torch.testing.assert_close(logprobs[0], expected)
    if topk:
        assert len(output_topk_logprobs) == 1
        row = output_topk_logprobs[0]
        assert row.batch_index == 0
        assert row.token_ids.tolist() == list(range(req.vocab_size - 1, req.vocab_size - 1 - topk, -1))
        torch.testing.assert_close(row.logprobs, reference[0, row.token_ids].cpu())
    else:
        assert output_topk_logprobs == [None]


def test_sample_candidate_logprobs_follow_mtp_row_selection(compiler, sampling_manager, monkeypatch):
    req = sampling_req(compiler, sampling_manager, req_idx=0, regular_constraint=None)
    ordinary = sampling_req(compiler, sampling_manager, req_idx=1, regular_constraint=None)
    tail = sampling_req(compiler, sampling_manager, req_idx=2, regular_constraint=None)
    for request in (req, ordinary, tail):
        request.sampling_param.shm_param.top_k = request.vocab_size
        request.sampling_param.shm_param.output_top_logprobs = 3
    ordinary.sampling_param.shm_param.output_top_logprobs = -1
    tail.sampling_param.shm_param.output_top_logprobs = 2
    logits = torch.arange(req.vocab_size, dtype=torch.float32, device="cuda").repeat(4, 1) / 10
    logits[3, :-2] = -torch.inf
    token_ids = torch.tensor([256, 0, 10, 256], device="cuda")
    monkeypatch.setattr(sampling, "_random_sample", lambda *args: token_ids)
    ids, logprobs, output_topk_logprobs = sampling.sample(logits, [req, req, ordinary, tail], [256])
    assert [None if row is None else row.batch_index for row in output_topk_logprobs] == [0, 1, None, 3]
    assert [row.token_ids.numel() for row in output_topk_logprobs if row is not None] == [3, 3, 2]
    backend = SimpleNamespace(args=SimpleNamespace(enable_rl=False), eos_id=[256], is_master_in_dp=True)
    ids_cpu, logprobs_cpu, ranks_cpu = ModeBackend._async_copy_next_token_infos_to_pin_mem(backend, ids, logprobs, ids)
    torch.cuda.current_stream().synchronize()
    # MTP drops row 1 and keeps an ordinary request with logprobs disabled in row 2.
    keep = torch.tensor([True, False, True, True])
    packs = [MagicMock() for _ in range(3)]
    monkeypatch.setattr(sampling_manager, "update_reqs_token_counter", MagicMock())
    ModeBackend._post_handle(
        backend,
        [req, ordinary, tail],
        ids_cpu[keep],
        logprobs_cpu[keep],
        ranks_cpu[keep],
        [row for row, selected in zip(output_topk_logprobs, keep.tolist()) if selected],
        packs,
    )
    first, middle, last = [pack.handle.call_args.kwargs for pack in packs]
    assert first["output_top_logprobs"][0] == [256, 255, 254]
    assert middle["output_top_logprobs"] is None
    assert last["output_top_logprobs"][0] == [256, 255]
    reference = logits.log_softmax(dim=-1).cpu()
    assert middle["next_token_logprob"] == pytest.approx(reference[2, 10].item())
    assert first["output_top_logprobs"][1] == pytest.approx(reference[0, [256, 255, 254]].tolist())
    assert last["output_top_logprobs"][1] == pytest.approx(reference[3, [256, 255]].tolist())
    assert logprobs_cpu[1].item() == -9999.0


@pytest.mark.parametrize("topk", [0, 1, 3, 20])
@pytest.mark.parametrize("rows", [[0, 1, 2, 3], [0, 2]])
@pytest.mark.parametrize("greedy", [False, True])
def test_sample_reuses_sampled_normalization(compiler, sampling_manager, monkeypatch, topk, rows, greedy):
    reqs = [sampling_req(compiler, sampling_manager, req_idx=i, regular_constraint=None) for i in range(4)]
    for req in reqs:
        req.sampling_param.shm_param.top_k = 1 if greedy else req.vocab_size
        req.sampling_param.shm_param.output_top_logprobs = topk
    logits = torch.arange(257, dtype=torch.float32, device="cuda").repeat(4, 1) / 10
    if not greedy:
        # Cover sampled tokens both inside and outside the requested top-k.
        ids = torch.tensor([256, 230, 0, 250], device="cuda")
        monkeypatch.setattr(sampling, "_random_sample", lambda *args: ids)
    ids, logprobs, output_topk_logprobs = sampling.sample(logits, reqs, [256], has_output=[i in rows for i in range(4)])
    reference = logits.log_softmax(dim=-1)
    topk_by_index = output_topk_logprobs
    for row in range(4):
        expected = reference[row, ids[row]]
        if row in rows and int((logits[row] > logits[row, ids[row]]).sum()) >= 20:
            expected = expected.new_tensor(-9999.0)
        torch.testing.assert_close(logprobs[row], expected)
        if topk and row in rows:
            top_ids, top_scores = topk_by_index[row].token_ids, topk_by_index[row].logprobs
            assert top_ids.tolist() == list(range(256, 256 - topk, -1))
            torch.testing.assert_close(top_scores, reference[row, top_ids].cpu())


def test_sample_preserves_sampled_score_and_keeps_ties(compiler, sampling_manager, monkeypatch):
    reqs = [sampling_req(compiler, sampling_manager, req_idx=i, regular_constraint=None) for i in range(4)]
    for req, topk in zip(reqs, [0, 0, 3, -1]):
        req.sampling_param.shm_param.top_k = req.vocab_size
        req.sampling_param.shm_param.output_top_logprobs = topk
    logits = torch.zeros(4, 257, device="cuda")
    logits[0, :19] = 1.0
    logits[1:, :20] = 1.0
    ids = torch.full((4,), 256, dtype=torch.int64, device="cuda")
    monkeypatch.setattr(sampling, "_random_sample", lambda *args: ids)
    _, logprobs, output_topk_logprobs = sampling.sample(logits, reqs, [256])
    assert [None if row is None else row.batch_index for row in output_topk_logprobs] == [None, None, 2, None]
    reference = logits.log_softmax(dim=-1)
    expected = reference[:, 256].clone()
    expected[1] = -9999.0
    expected[2] = -9999.0
    torch.testing.assert_close(logprobs, expected)
    candidate_ids, candidate_logprobs = output_topk_logprobs[2].token_ids, output_topk_logprobs[2].logprobs
    assert len(set(candidate_ids.tolist())) == 3 and (candidate_ids < 20).all()
    torch.testing.assert_close(candidate_logprobs, reference[2, candidate_ids].cpu())


@pytest.mark.parametrize("sampling_manager", ["cpu_counter", "pin_mem_counter", "gpu_counter"], indirect=True)
def test_sample_reads_scattered_prefix_after_existing_handoff(compiler, sampling_manager, monkeypatch):
    req = sampling_req(compiler, sampling_manager)
    table = sampling_manager.req_to_next_token_ids
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    # This is the existing previous-step completion event, before post_handle.
    previous_done = torch.cuda.Event()
    with torch.cuda.stream(stream):
        scatter_mtp_next_tokens(
            backend=SimpleNamespace(model=SimpleNamespace(req_manager=sampling.g_infer_context.req_manager)),
            proposal=SimpleNamespace(token_ids=torch.tensor([[ord("b"), ord("!")]], device="cuda")),
            b_req_mtp_start_loc=torch.tensor([0], dtype=torch.int32, device="cuda"),
            target_next_token_ids=torch.tensor([ord("a")], device="cuda"),
            b_req_idx=torch.tensor([req.req_idx], dtype=torch.int32, device="cuda"),
            mtp_accept_len=torch.tensor([1], dtype=torch.int32, device="cuda"),
        )
        previous_done.record()
    previous_done.synchronize()
    commit_token(req, ord("a"))

    def unexpected_copy(*args, **kwargs):
        pytest.fail("Sampling must read the ready CPU snapshot without starting another D2H")

    monkeypatch.setattr(sampling.g_pin_mem_manager, "async_copy_from_gpu_tensor", unexpected_copy)
    monkeypatch.setattr(sampling.g_pin_mem_manager, "async_copy_from_gpu_tensor_with_event", unexpected_copy)
    tensor_tolist = torch.Tensor.tolist

    def cpu_tolist(tensor):
        assert not tensor.is_cuda, "Sampling synchronized a GPU tensor to read the prefix"
        return tensor_tolist(tensor)

    logits = torch.zeros(3, 257, device="cuda")
    logits[:, ord("!")] = 10
    # Three selected verify rows use a prefix of the four-column request table.
    with monkeypatch.context() as patcher:
        patcher.setattr(torch.Tensor, "tolist", cpu_tolist)
        next_tokens, logprobs, _ = sampling.sample(
            logits, [req] * 3, [256], b_mtp_index=torch.arange(3, dtype=torch.int32, device="cuda")
        )
    assert next_tokens.tolist() == [ord("b"), ord("c"), ord("!")]
    assert torch.isfinite(logprobs).all()
    assert allowed([req])[0].nonzero().flatten().tolist() == [ord("b")]

    accept_len, accepted = mtp_verify(
        table,
        torch.tensor([0], dtype=torch.int32, device="cuda"),
        next_tokens,
        torch.full((3,), req.req_idx, dtype=torch.int32, device="cuda"),
    )
    assert accept_len.is_cuda and accepted.is_cuda
    assert accept_len.tolist() == [2] and accepted.tolist() == [1, 1, 0]
    assert table[req.req_idx].tolist() == [ord("a"), ord("b"), ord("!"), 1]


def test_sample_does_not_activate_constraints_for_partial_prefill(compiler, sampling_manager):
    req = sampling_req(compiler, sampling_manager)
    req.output_constraint.fill_masks = MagicMock(side_effect=AssertionError("Partial prefill must not fill masks"))
    logits = torch.zeros(1, 257, device="cuda")
    logits[:, ord("!")] = 10

    next_tokens, logprobs, _ = sampling.sample(logits, [req], [256], has_output=[False])

    assert next_tokens.tolist() == [ord("!")]
    assert torch.isfinite(logprobs).all()
    req.output_constraint.fill_masks.assert_not_called()
    assert req.output_constraint.error is None


@pytest.mark.parametrize("failure", ["fill_next_token_bitmask", "rollback"])
def test_sample_preserves_valid_masks_and_isolates_failed_requests(compiler, sampling_manager, monkeypatch, failure):
    import xgrammar as xgr

    bad = sampling_req(compiler, sampling_manager, req_idx=0)
    good = sampling_req(compiler, sampling_manager, req_idx=1, regular_constraint="cd")
    bad_matcher = bad.output_constraint.matcher
    original = getattr(xgr.GrammarMatcher, failure)

    def fail_for_bad_request(matcher, *args):
        if matcher is bad_matcher and (failure == "rollback" or args[1] == 1):
            if failure == "fill_next_token_bitmask":
                args[0].zero_()
            raise RuntimeError("injected grammar failure")
        return original(matcher, *args)

    monkeypatch.setattr(xgr.GrammarMatcher, failure, fail_for_bad_request)

    sampling_manager.req_to_next_token_ids[bad.req_idx] = torch.tensor([0, *b"ab!"])
    copy_snapshot_and_wait(sampling_manager)
    sampling_manager.req_to_bitmask_enabled[bad.req_idx] = True
    sampling_manager.req_to_bitmask[bad.req_idx].zero_()
    logits = torch.zeros(3, 257, device="cuda")
    logits[:, ord("!")] = 10
    tokens, logprobs, _ = sampling.sample(
        logits,
        [bad, bad, good],
        [256],
        b_mtp_index=torch.tensor([0, 1, 0], dtype=torch.int32, device="cuda"),
    )

    expected_tokens = b"abc" if failure == "rollback" else b"!!c"
    assert tokens.tolist() == list(expected_tokens)
    assert torch.isfinite(logprobs).all()
    assert bool(sampling_manager.req_to_bitmask_enabled[bad.req_idx]) == (failure == "rollback")
    assert sampling_manager.req_to_bitmask_enabled[good.req_idx]
    assert bad.output_constraint.error == "injected grammar failure"
    assert good.output_constraint.error is None


def test_sample_groups_mixed_verify_rows_and_preserves_speculative_reasoning(compiler, sampling_manager):
    active = sampling_req(compiler, sampling_manager, req_idx=0)
    thinking = sampling_req(compiler, sampling_manager, req_idx=1, guided_reasoning_end=tuple(b"]>"))
    plain = sampling_req(compiler, sampling_manager, req_idx=2, regular_constraint=None)
    partial = sampling_req(compiler, sampling_manager, req_idx=3)
    commit_token(active, ord("a"))
    commit_token(thinking, ord("]"))
    table = sampling_manager.req_to_next_token_ids
    table[active.req_idx] = torch.tensor(list(b"ab!!"))
    table[thinking.req_idx] = torch.tensor(list(b"]>a!"))
    copy_snapshot_and_wait(sampling_manager)

    partial.output_constraint.fill_masks = MagicMock(side_effect=AssertionError("Partial prefill must not fill masks"))
    # Dynamic verification can keep a different prefix length per request.
    reqs = [plain, active, active, thinking, thinking, thinking, plain, partial]
    logits = torch.zeros(len(reqs), 257, device="cuda")
    logits[:, ord("!")] = 10
    next_tokens, logprobs, _ = sampling.sample(
        logits,
        reqs,
        [256],
        has_output=[True] * 7 + [False],
        b_mtp_index=torch.tensor([0, 0, 1, 0, 1, 2, 0, 0], dtype=torch.int32, device="cuda"),
    )

    assert next_tokens.tolist() == list(b"!bc!ab!!")
    assert torch.isfinite(logprobs).all()
    assert allowed([active])[0].nonzero().flatten().tolist() == [ord("b")]
    assert thinking.output_constraint.in_reasoning
    assert thinking.output_constraint.reasoning_tail == (ord("]"),)
    partial.output_constraint.fill_masks.assert_not_called()
    assert partial.output_constraint.error is None
    assert not commit_token(thinking, ord(">"))
    assert allowed([thinking])[0].nonzero().flatten().tolist() == [ord("a")]


def test_reused_mask_slots_clear_an_unreachable_draft_suffix(compiler, sampling_manager):
    req = sampling_req(compiler, sampling_manager)
    positions = torch.arange(4, dtype=torch.int32, device="cuda")
    sampling_manager.req_to_next_token_ids[req.req_idx] = torch.tensor([0, *b"abc"])
    copy_snapshot_and_wait(sampling_manager)
    logits = torch.zeros(4, 257, device="cuda")
    logits[:, ord("!")] = 10
    tokens, _, _ = sampling.sample(logits, [req] * 4, [256], b_mtp_index=positions)
    assert tokens.tolist() == [*b"abc", 256]
    completed = torch.cuda.Event()
    completed.record()
    completed.synchronize()

    # The second prefix rejects its draft at '!', so masks from the previous
    # iteration must not constrain that unreachable suffix.
    sampling_manager.req_to_next_token_ids[req.req_idx] = torch.tensor([0, *b"a!c"])
    copy_snapshot_and_wait(sampling_manager)
    logits.zero_()
    logits[:, ord("!")] = 10
    tokens, _, _ = sampling.sample(logits, [req] * 4, [256], b_mtp_index=positions)
    assert tokens.tolist() == list(b"ab!!")
    assert allowed([req])[0].nonzero().flatten().tolist() == [ord("a")]


@pytest.mark.parametrize("replacement", ["plain", "partial", "thinking", "finished"])
def test_reused_request_slot_disables_stale_mask(compiler, sampling_manager, replacement):
    from lightllm.server.core.objs import FinishStatus

    old = sampling_req(compiler, sampling_manager, req_idx=0)
    tokens, _, _ = sampling.sample(torch.zeros(1, 257, device="cuda"), [old], [256])
    assert tokens.tolist() == [ord("a")]
    completed = torch.cuda.Event()
    completed.record()
    completed.synchronize()
    assert sampling_manager.req_to_bitmask_enabled[0]

    constraints = {"regular_constraint": None} if replacement == "plain" else {}
    if replacement == "thinking":
        constraints["guided_reasoning_end"] = (ord("]"),)
    new = sampling_req(compiler, sampling_manager, req_idx=0, **constraints)
    if replacement == "finished":
        new.finish_status.set_status(FinishStatus.FINISHED_STOP)
    good = sampling_req(compiler, sampling_manager, req_idx=1, regular_constraint="xy")
    logits = torch.zeros(2, 257, device="cuda")
    logits[:, ord("!")] = 10
    tokens, _, _ = sampling.sample(logits, [new, good], [256], has_output=[replacement != "partial", True])

    assert tokens.tolist() == list(b"!x")
    assert not sampling_manager.req_to_bitmask_enabled[0]
    assert sampling_manager.req_to_bitmask_enabled[1]


@pytest.mark.parametrize("constraints_enabled,draft_step", [(True, 2), (False, 2), (True, 0)])
def test_mtp_scatter_reuses_snapshot_and_only_copies_when_needed(sampling_manager, constraints_enabled, draft_step):
    table = sampling_manager.req_to_next_token_ids
    snapshot = sampling_manager.req_to_next_token_ids_cpu
    snapshot.fill_(-42)
    snapshot_ptr = snapshot.data_ptr()
    if not constraints_enabled:
        sampling_manager.req_to_bitmask_enabled = None
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    backend = SimpleNamespace(model=SimpleNamespace(req_manager=sampling.g_infer_context.req_manager))
    for first_draft in [10, 20]:
        with torch.cuda.stream(stream):
            drafts = torch.arange(first_draft, first_draft + draft_step, device="cuda").reshape(1, draft_step)
            scatter_mtp_next_tokens(
                backend=backend,
                proposal=SimpleNamespace(token_ids=drafts),
                target_next_token_ids=torch.tensor([7], device="cuda"),
                b_req_mtp_start_loc=torch.tensor([0], dtype=torch.int32, device="cuda"),
                b_req_idx=torch.tensor([0], dtype=torch.int32, device="cuda"),
                mtp_accept_len=torch.tensor([1], dtype=torch.int32, device="cuda"),
            )
            completed = torch.cuda.Event()
            completed.record()
        completed.synchronize()
        assert snapshot.data_ptr() == snapshot_ptr
        if constraints_enabled:
            expected = [7, first_draft, first_draft + 1, 1] if draft_step else [7, 1, 1, 1]
            assert snapshot[0].tolist() == expected
            # CPU snapshot is independent of subsequent writes to the GPU table.
            table[0].fill_(99)
            torch.cuda.current_stream().synchronize()
            assert snapshot[0].tolist() == expected
        else:
            assert snapshot.eq(-42).all()


def copy_snapshot_and_wait(manager):
    manager.req_to_next_token_ids_cpu.copy_(manager.req_to_next_token_ids, non_blocking=True)
    completed = torch.cuda.Event()
    completed.record()
    completed.synchronize()


@pytest.mark.parametrize("vocab_size", [30, 2051, 32769])
def test_capture_across_vocab_tiles_preserves_sampled_scores(vocab_size):
    logits = torch.linspace(3.0, 0.1, vocab_size, device="cuda").repeat(4, 1)
    ids = torch.tensor([1, 19, 20, vocab_size - 1], device="cuda")
    logprobs = logits.log_softmax(-1).gather(1, ids[:, None]).squeeze(1)
    original = logprobs.clone()
    rows = torch.tensor([[0, 0, 0], [1, 1, 0], [2, 3, 1], [3, 20, 4]], device="cuda", dtype=torch.int32)
    candidate_ids, scores = sampling.capture_output_logprobs(logits, ids, logprobs, rows, 20, 24)
    torch.testing.assert_close(logprobs[:2], original[:2], rtol=0, atol=0)
    assert logprobs[2].item() == -9999.0 and logprobs[3].item() == -9999.0
    for row, width, offset in rows.tolist():
        assert candidate_ids[offset : offset + width].tolist() == list(range(width))
        torch.testing.assert_close(scores[offset : offset + width], logits.log_softmax(-1)[row, :width])
