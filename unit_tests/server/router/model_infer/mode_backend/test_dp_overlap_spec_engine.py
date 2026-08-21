from types import SimpleNamespace

import pytest
import torch

from lightllm.server.router.model_infer.mode_backend.dp_backend import impl as dp_backend_impl
from lightllm.server.router.model_infer.mode_backend.dp_backend.impl import DPChunkedPrefillBackend
from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend
from lightllm.server.router.model_infer.mode_backend.chunked_prefill.impl import ChunkedPrefillBackend
from lightllm.server.router.model_infer.mtp_speculative.dp_overlap_engine import DPOverlapSpecEngine
from lightllm.server.router.model_infer.mtp_speculative.engine import SpecEngine
from lightllm.server.router.model_infer.mtp_speculative.planner import FixedSpecPlanner, SpecDecodePlan
from lightllm.server.router.model_infer.mtp_speculative.proposers.base import (
    MtpMemIndexesToFree,
    SpecProposal,
)


class _RecordingSpecEngine:
    def __init__(self):
        self.propose_args = None
        self.propose_overlap_args = None

    def propose_next(self, **kwargs):
        self.propose_args = kwargs
        token_ids = kwargs["target_next_token_ids"].new_zeros((kwargs["b_req_mtp_start_loc"].shape[0], 7))
        return SpecProposal(
            token_ids=token_ids,
            extra_mem_indexes_cpu=[MtpMemIndexesToFree(mem_indexes_cpu=torch.tensor([123], dtype=torch.int32))],
        )

    def propose_next_overlap(self, **kwargs):
        self.propose_overlap_args = kwargs
        request_count = (kwargs["real_verify_rows0"] + kwargs["real_verify_rows1"]) // 8
        token_ids = kwargs["target_next_token_ids0"].new_zeros((request_count, 7))
        return SpecProposal(
            token_ids=token_ids,
            extra_mem_indexes_cpu=[MtpMemIndexesToFree(mem_indexes_cpu=torch.tensor([456], dtype=torch.int32))],
        )


def _capture_scatter_args(monkeypatch):
    scatter_args = {}
    monkeypatch.setattr(
        dp_backend_impl.mtp_utils, "scatter_mtp_next_tokens", lambda **kwargs: scatter_args.update(kwargs)
    )
    return scatter_args


def _assert_all_mem_indexes_are_freed(extra_mem_indexes_cpu, expected):
    assert len(extra_mem_indexes_cpu) == 1
    assert torch.equal(extra_mem_indexes_cpu[0].mem_indexes_cpu, expected)
    assert extra_mem_indexes_cpu[0].free_mask_cpu is None


def test_dp_backend_reuses_common_engine_outside_overlap():
    args = SimpleNamespace(mtp_mode="eagle3", mtp_dynamic_verify=False)
    backend = ChunkedPrefillBackend.__new__(ChunkedPrefillBackend)
    backend.args = args
    backend.max_draft_step = 2
    backend.init_spec_engine()
    dp_backend = DPChunkedPrefillBackend.__new__(DPChunkedPrefillBackend)
    dp_backend.args = args
    dp_backend.max_draft_step = 2
    dp_backend.enable_prefill_microbatch_overlap = False
    dp_backend.enable_decode_microbatch_overlap = True
    dp_backend.init_spec_engine()

    assert "spec_engine_class" not in ModeBackend.__dict__
    assert type(backend.spec_engine) is SpecEngine
    assert type(dp_backend.spec_engine) is SpecEngine
    assert dp_backend.spec_engine.allow_empty_batch
    assert not dp_backend.spec_engine.proposer.enable_dynmaic_mtp
    assert type(dp_backend.spec_engine.planner) is FixedSpecPlanner
    assert type(dp_backend.dp_overlap_spec_engine) is DPOverlapSpecEngine
    assert dp_backend.prefill_draft_engine is dp_backend.spec_engine
    assert dp_backend.decode_draft_engine is dp_backend.dp_overlap_spec_engine
    assert not issubclass(DPOverlapSpecEngine, SpecEngine)


def test_dp_prefill_and_decode_select_overlap_engine_independently():
    args = SimpleNamespace(mtp_mode="eagle3", mtp_dynamic_verify=False)

    for prefill_overlap in (False, True):
        for decode_overlap in (False, True):
            backend = DPChunkedPrefillBackend.__new__(DPChunkedPrefillBackend)
            backend.args = args
            backend.max_draft_step = 2
            backend.enable_prefill_microbatch_overlap = prefill_overlap
            backend.enable_decode_microbatch_overlap = decode_overlap
            backend.init_spec_engine()

            expected_prefill_engine = backend.dp_overlap_spec_engine if prefill_overlap else backend.spec_engine
            expected_decode_engine = backend.dp_overlap_spec_engine if decode_overlap else backend.spec_engine
            assert backend.prefill_draft_engine is expected_prefill_engine
            assert backend.decode_draft_engine is expected_decode_engine


def test_padded_token_ids_support_empty_dp_rank():
    padded_token_ids = DPChunkedPrefillBackend._build_padded_next_token_ids(
        token_ids=None,
        batch_size=4,
        copy_len=0,
        device=torch.device("cpu"),
    )

    assert torch.equal(padded_token_ids, torch.zeros(4, dtype=torch.int64))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_dp_decode_mtp_runs_common_engine_for_empty_batch(monkeypatch):
    device = "cuda"
    empty_i32 = torch.empty((0,), dtype=torch.int32, device=device)
    model_input = SimpleNamespace(
        batch_size=0,
        b_req_idx=empty_i32,
        b_mtp_index=empty_i32,
        mem_indexes_cpu=torch.empty((0,), dtype=torch.int32),
    )
    model_output = SimpleNamespace(logits=torch.empty((0, 8), device=device))
    calls = []

    class _CommonEngine:
        def plan_decode(self, **kwargs):
            calls.append("plan")
            return SpecDecodePlan(0, 0, 2, 2)

        def prepare_decode_model_input(self, **kwargs):
            calls.append("prepare")
            return kwargs["model_input"], None

        def propose_next(self, **kwargs):
            calls.append("propose")
            assert kwargs["target_next_token_ids"].shape == (0,)
            assert kwargs["b_req_mtp_start_loc"].shape == (0,)
            assert kwargs["accept_len"].shape == (0,)
            return SpecProposal(token_ids=torch.empty((0, 2), dtype=torch.int64, device=device))

    backend = DPChunkedPrefillBackend.__new__(DPChunkedPrefillBackend)
    backend.spec_engine = _CommonEngine()
    backend.model = SimpleNamespace(forward=lambda _: model_output)
    event_pack = SimpleNamespace(
        notify_post_handle_and_wait_pre_post_handle=lambda: calls.append("post_wait"),
        notify_forward_and_wait_post_handle=lambda: calls.append("forward_wait"),
        notify_pre_post_handle=lambda: calls.append("pre_post"),
    )
    monkeypatch.setattr(
        dp_backend_impl,
        "prepare_decode_inputs",
        lambda req_objs: (model_input, []),
    )
    monkeypatch.setattr(
        dp_backend_impl,
        "g_infer_context",
        SimpleNamespace(get_overlap_stream=lambda: torch.cuda.current_stream()),
    )
    monkeypatch.setattr(
        dp_backend_impl.mtp_utils,
        "free_mem_indexes",
        lambda **kwargs: calls.append("free"),
    )

    backend.decode_mtp(event_pack=event_pack, decode_reqs=[])

    assert calls == ["plan", "prepare", "propose", "post_wait", "forward_wait", "free", "pre_post"]


def test_dp_overlap_eagle_passes_both_fixed_verify_layouts_to_proposer(monkeypatch):
    scatter_args = _capture_scatter_args(monkeypatch)
    backend = DPChunkedPrefillBackend.__new__(DPChunkedPrefillBackend)
    backend.max_draft_step = 7
    backend.spec_engine = _RecordingSpecEngine()
    backend.dp_overlap_spec_engine = _RecordingSpecEngine()
    backend.decode_draft_engine = backend.dp_overlap_spec_engine
    model_input0 = SimpleNamespace(
        batch_size=16,
        b_req_idx=torch.arange(16, dtype=torch.int32),
    )
    model_input1 = SimpleNamespace(
        batch_size=16,
        b_req_idx=torch.arange(16, dtype=torch.int32),
    )
    model_output0 = SimpleNamespace(spec_hidden=torch.randn(16, 4))
    model_output1 = SimpleNamespace(spec_hidden=torch.randn(16, 4))
    next_token_ids = torch.arange(24, dtype=torch.int64)
    b_req_idx = torch.arange(24, dtype=torch.int32)
    start_locs = torch.tensor([0, 8, 16], dtype=torch.int32)
    accept_len = torch.tensor([2, 3, 4], dtype=torch.int32)

    extra_mem = backend._draft_decode_eagle_overlap(
        model_input0=model_input0,
        model_output0=model_output0,
        model_input1=model_input1,
        model_output1=model_output1,
        b_req_idx=b_req_idx,
        next_token_ids=next_token_ids,
        mtp_accept_len=accept_len,
        b_req_mtp_start_loc=start_locs,
        req_num0=8,
        req_num1=16,
    )

    propose_args = backend.dp_overlap_spec_engine.propose_overlap_args
    assert propose_args["target_next_token_ids0"].shape == (16,)
    assert propose_args["target_next_token_ids1"].shape == (16,)
    assert propose_args["real_verify_rows0"] == 8
    assert propose_args["real_verify_rows1"] == 16
    assert torch.equal(propose_args["accept_len0"], torch.tensor([2, 1], dtype=torch.int32))
    assert torch.equal(propose_args["accept_len1"], torch.tensor([3, 4], dtype=torch.int32))
    assert scatter_args["proposal"].token_ids.shape == (3, 7)
    assert torch.equal(scatter_args["target_next_token_ids"], next_token_ids)
    _assert_all_mem_indexes_are_freed(extra_mem, torch.tensor([456], dtype=torch.int32))


def test_dp_overlap_vanilla_delegates_both_microbatches_to_proposer(monkeypatch):
    scatter_args = _capture_scatter_args(monkeypatch)
    backend = DPChunkedPrefillBackend.__new__(DPChunkedPrefillBackend)
    backend.max_draft_step = 7
    backend.spec_engine = _RecordingSpecEngine()
    backend.dp_overlap_spec_engine = _RecordingSpecEngine()
    backend.decode_draft_engine = backend.dp_overlap_spec_engine
    model_input0 = SimpleNamespace(
        batch_size=16,
        b_req_idx=torch.arange(16, dtype=torch.int32),
    )
    model_input1 = SimpleNamespace(
        batch_size=16,
        b_req_idx=torch.arange(16, dtype=torch.int32),
    )
    next_token_ids = torch.arange(24, dtype=torch.int64)

    extra_mem = backend._draft_decode_vanilla_overlap(
        model_input0=model_input0,
        model_output0=SimpleNamespace(),
        model_input1=model_input1,
        model_output1=SimpleNamespace(),
        b_req_idx=torch.arange(24, dtype=torch.int32),
        next_token_ids=next_token_ids,
        mtp_accept_len=torch.tensor([2, 3, 4], dtype=torch.int32),
        b_req_mtp_start_loc=torch.tensor([0, 8, 16], dtype=torch.int32),
        req_num0=8,
        req_num1=16,
    )

    propose_args = backend.dp_overlap_spec_engine.propose_overlap_args
    assert propose_args["target_next_token_ids0"].shape == (16,)
    assert propose_args["target_next_token_ids1"].shape == (16,)
    assert propose_args["real_verify_rows0"] == 8
    assert propose_args["real_verify_rows1"] == 16
    assert torch.equal(propose_args["accept_len0"], torch.tensor([2], dtype=torch.int32))
    assert torch.equal(propose_args["accept_len1"], torch.tensor([3, 4], dtype=torch.int32))
    assert scatter_args["proposal"].token_ids.shape == (3, 7)
    assert torch.equal(scatter_args["target_next_token_ids"], next_token_ids)
    _assert_all_mem_indexes_are_freed(extra_mem, torch.tensor([456], dtype=torch.int32))
