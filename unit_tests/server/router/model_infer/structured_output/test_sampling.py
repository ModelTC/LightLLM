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
from lightllm.common.basemodel.triton_kernel.mtp_utils import mtp_scatter_next_token_ids, mtp_verify
from lightllm.server.router.model_infer.mode_backend import generic_post_process as sampling


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
    assert result.req_to_next_token_ids.device.type == "cpu" and result.req_to_next_token_ids.is_pinned()
    assert result.req_to_next_token_scores.is_cuda
    monkeypatch.setattr(sampling.g_infer_context, "req_manager", SimpleNamespace(req_sampling_params_manager=result))
    yield result
    torch.cuda.synchronize()


def sampling_req(compiler, sampling_manager, req_idx=5, **constraints):
    req = make_req(**({"regular_constraint": "abc"} | constraints))
    req.req_idx = req_idx
    req.sampling_param.shm_param = SimpleNamespace(
        presence_penalty=0.0,
        frequency_penalty=0.0,
        repetition_penalty=1.0,
        exponential_decay_length_penalty=SimpleNamespace(to_tuple=lambda: (0, 1.1)),
        input_penalty=False,
        min_new_tokens=0,
        temperature=1.0,
        top_p=1.0,
        top_k=1,
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
    next_tokens, logprobs = sampling.sample(logits, [req], [256])

    assert next_tokens.tolist() == [ord("!")]
    assert torch.isfinite(logprobs).all()
    assert not sampling_manager.req_to_bitmask_enabled[req.req_idx]
    assert not compiler._cache


@pytest.mark.parametrize("sampling_manager", ["cpu_counter", "pin_mem_counter", "gpu_counter"], indirect=True)
def test_sample_reads_scattered_prefix_after_existing_handoff(compiler, sampling_manager, monkeypatch):
    req = sampling_req(compiler, sampling_manager)
    table = sampling_manager.req_to_next_token_ids
    mtp_scatter_next_token_ids(
        req_to_next_token_ids=table,
        b_req_mtp_start_loc=torch.tensor([0], dtype=torch.int32, device="cuda"),
        target_next_token_ids=torch.tensor([ord("a")], device="cuda"),
        draft_token_ids=torch.tensor([[ord("b"), ord("!")]], device="cuda"),
        b_req_idx=torch.tensor([req.req_idx], dtype=torch.int32, device="cuda"),
        mtp_accept_len=torch.tensor([1], dtype=torch.int32, device="cuda"),
    )
    # This is the existing previous-step completion event, before post_handle.
    previous_done = torch.cuda.Event()
    previous_done.record()
    previous_done.synchronize()
    commit_token(req, ord("a"))

    def unexpected_copy(*args, **kwargs):
        pytest.fail("Sampling must read the ready pinned table without a prefix D2H")

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
        next_tokens, logprobs = sampling.sample(
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

    next_tokens, logprobs = sampling.sample(logits, [req], [256], has_output=[False])

    assert next_tokens.tolist() == [ord("!")]
    assert torch.isfinite(logprobs).all()
    req.output_constraint.fill_masks.assert_not_called()
    assert req.output_constraint.error is None


@pytest.mark.parametrize("failure", ["fill_next_token_bitmask", "rollback"])
def test_sample_ignores_failed_masks_and_preserves_other_requests(compiler, sampling_manager, monkeypatch, failure):
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
    sampling_manager.req_to_bitmask_enabled[bad.req_idx] = True
    sampling_manager.req_to_bitmask[bad.req_idx].zero_()
    logits = torch.zeros(3, 257, device="cuda")
    logits[:, ord("!")] = 10
    tokens, logprobs = sampling.sample(
        logits,
        [bad, bad, good],
        [256],
        b_mtp_index=torch.tensor([0, 1, 0], dtype=torch.int32, device="cuda"),
    )

    assert tokens.tolist() == list(b"!!c")
    assert torch.isfinite(logprobs).all()
    assert not sampling_manager.req_to_bitmask_enabled[bad.req_idx]
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

    partial.output_constraint.fill_masks = MagicMock(side_effect=AssertionError("Partial prefill must not fill masks"))
    # Dynamic verification can keep a different prefix length per request.
    reqs = [plain, active, active, thinking, thinking, thinking, plain, partial]
    logits = torch.zeros(len(reqs), 257, device="cuda")
    logits[:, ord("!")] = 10
    next_tokens, logprobs = sampling.sample(
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
    logits = torch.zeros(4, 257, device="cuda")
    logits[:, ord("!")] = 10
    tokens, _ = sampling.sample(logits, [req] * 4, [256], b_mtp_index=positions)
    assert tokens.tolist() == [*b"abc", 256]
    completed = torch.cuda.Event()
    completed.record()
    completed.synchronize()

    # The second prefix rejects its draft at '!', so masks from the previous
    # iteration must not constrain that unreachable suffix.
    sampling_manager.req_to_next_token_ids[req.req_idx] = torch.tensor([0, *b"a!c"])
    logits.zero_()
    logits[:, ord("!")] = 10
    tokens, _ = sampling.sample(logits, [req] * 4, [256], b_mtp_index=positions)
    assert tokens.tolist() == list(b"ab!!")
    assert allowed([req])[0].nonzero().flatten().tolist() == [ord("a")]


@pytest.mark.parametrize("replacement", ["plain", "partial", "thinking", "finished"])
def test_reused_request_slot_disables_stale_mask(compiler, sampling_manager, replacement):
    from lightllm.server.core.objs import FinishStatus

    old = sampling_req(compiler, sampling_manager, req_idx=0)
    tokens, _ = sampling.sample(torch.zeros(1, 257, device="cuda"), [old], [256])
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
    tokens, _ = sampling.sample(logits, [new, good], [256], has_output=[replacement != "partial", True])

    assert tokens.tolist() == list(b"!x")
    assert not sampling_manager.req_to_bitmask_enabled[0]
    assert sampling_manager.req_to_bitmask_enabled[1]
