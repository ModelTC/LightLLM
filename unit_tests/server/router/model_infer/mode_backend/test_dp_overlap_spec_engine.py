from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from lightllm.server.router.model_infer.mode_backend.dp_backend import (
    impl as dp_backend_impl,
)
from lightllm.server.router.model_infer.mode_backend.dp_backend.impl import (
    DPChunkedPrefillBackend,
)
from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend
from lightllm.server.router.model_infer.mode_backend.chunked_prefill.impl import (
    ChunkedPrefillBackend,
)
from lightllm.server.router.model_infer.mtp_speculative.dp_overlap_engine import (
    DPOverlapSpecEngine,
)
from lightllm.server.router.model_infer.mtp_speculative.engine import SpecEngine
from lightllm.server.router.model_infer.mtp_speculative.planner import (
    lightspec as lightspec_planner_impl,
)
from lightllm.server.router.model_infer.mtp_speculative.planner import (
    FixedSpecPlanner,
    LightSpecPlanner,
    SpecDecodePlan,
)
from lightllm.server.router.model_infer.mtp_speculative.proposers.base import (
    SpecProposal,
)


def test_dp_backend_reuses_common_engine_outside_overlap():
    args = SimpleNamespace(mtp_mode="eagle3", mtp_dynamic_verify=False, dp=1)
    backend = ChunkedPrefillBackend.__new__(ChunkedPrefillBackend)
    backend.args = args
    backend.max_draft_step = 2
    backend.init_spec_engine()
    dp_backend = DPChunkedPrefillBackend.__new__(DPChunkedPrefillBackend)
    dp_backend.args = args
    dp_backend.max_draft_step = 2
    dp_backend.dp_size = 1
    dp_backend.enable_prefill_microbatch_overlap = False
    dp_backend.enable_decode_microbatch_overlap = True
    dp_backend.init_spec_engine()

    assert "spec_engine_class" not in ModeBackend.__dict__
    assert type(backend.spec_engine) is SpecEngine
    assert type(dp_backend.spec_engine) is SpecEngine
    assert not dp_backend.spec_engine.proposer.enable_dynmaic_mtp
    assert type(dp_backend.spec_engine.planner) is FixedSpecPlanner
    assert type(dp_backend.dp_overlap_spec_engine) is DPOverlapSpecEngine
    assert dp_backend.dp_overlap_spec_engine.common_engine is dp_backend.spec_engine
    assert dp_backend.prefill_draft_engine is dp_backend.spec_engine
    assert dp_backend.decode_draft_engine is dp_backend.dp_overlap_spec_engine
    assert not issubclass(DPOverlapSpecEngine, SpecEngine)


def test_lightspec_planner_reduces_draft_step_with_max(monkeypatch):
    group = object()
    planner = LightSpecPlanner.__new__(LightSpecPlanner)
    planner._draft_step_group = group
    planner._draft_step_tensor = torch.zeros((1,), dtype=torch.int32)
    planner._draft_step_stream = object()
    planner.draft_steps = (1, 2, 3)
    planner.pre_draft_step = 1
    planner.backend = SimpleNamespace(dp_size=2)

    def all_reduce(tensor, op, group, async_op):
        assert op == torch.distributed.ReduceOp.MAX
        assert group is planner._draft_step_group
        assert not async_op
        tensor.fill_(3)

    monkeypatch.setattr(lightspec_planner_impl.dist, "all_reduce", all_reduce)
    monkeypatch.setattr(lightspec_planner_impl.torch.cuda, "stream", lambda stream: nullcontext())

    plan = planner.plan(decode_reqs=[], origin_batch_size=2)

    assert plan.dynamic_batch_size == 2
    assert plan.draft_step == 3
    assert planner.pre_draft_step == 3


def test_dp_backend_builds_dedicated_global_nccl_group(monkeypatch):
    created_group = object()
    created_stream = object()
    new_group_args = {}
    real_torch_zeros = torch.zeros

    def new_group(*, ranks, backend):
        new_group_args.update(ranks=ranks, backend=backend)
        return created_group

    def zeros_without_cuda(*args, **kwargs):
        kwargs.pop("device", None)
        return real_torch_zeros(*args, **kwargs)

    monkeypatch.setattr(lightspec_planner_impl, "get_global_world_size", lambda: 4)
    monkeypatch.setattr(lightspec_planner_impl.dist, "new_group", new_group)
    monkeypatch.setattr(lightspec_planner_impl.torch, "zeros", zeros_without_cuda)
    monkeypatch.setattr(lightspec_planner_impl.torch.cuda, "Stream", lambda: created_stream)

    backend = DPChunkedPrefillBackend.__new__(DPChunkedPrefillBackend)
    backend.args = SimpleNamespace(mtp_mode="eagle3", mtp_dynamic_verify=True, dp=2)
    backend.max_draft_step = 3
    backend.dp_size = 2
    backend.model = SimpleNamespace(graph=None)
    backend.draft_models = [SimpleNamespace(graph=None)]
    backend.enable_prefill_microbatch_overlap = False
    backend.enable_decode_microbatch_overlap = False
    backend.init_spec_engine()

    assert new_group_args == {"ranks": [0, 1, 2, 3], "backend": "nccl"}
    planner = backend.spec_engine.planner
    assert planner._draft_step_group is created_group
    assert planner._draft_step_tensor.shape == (1,)
    assert planner._draft_step_stream is created_stream


def test_dp_backend_does_not_build_group_for_single_draft_step_mode(monkeypatch):
    monkeypatch.setattr(
        lightspec_planner_impl.dist,
        "new_group",
        lambda **kwargs: pytest.fail(f"unexpected NCCL group: {kwargs}"),
    )

    backend = DPChunkedPrefillBackend.__new__(DPChunkedPrefillBackend)
    backend.args = SimpleNamespace(mtp_mode="vanilla_with_att", mtp_dynamic_verify=True, dp=2)
    backend.max_draft_step = 3
    backend.dp_size = 2
    backend.model = SimpleNamespace(graph=None)
    backend.draft_models = [SimpleNamespace(graph=None, block_size=4)]
    backend.enable_prefill_microbatch_overlap = False
    backend.enable_decode_microbatch_overlap = False

    backend.init_spec_engine()

    assert backend.spec_engine.planner._draft_step_group is None
    assert backend.spec_engine.planner._draft_step_tensor is None
    assert backend.spec_engine.planner._draft_step_stream is None


def test_dp_backend_does_not_build_group_for_fixed_planner(monkeypatch):
    monkeypatch.setattr(
        lightspec_planner_impl.dist,
        "new_group",
        lambda **kwargs: pytest.fail(f"unexpected NCCL group: {kwargs}"),
    )

    backend = DPChunkedPrefillBackend.__new__(DPChunkedPrefillBackend)
    backend.args = SimpleNamespace(mtp_mode="eagle3", mtp_dynamic_verify=False, dp=2)
    backend.max_draft_step = 3
    backend.dp_size = 2
    backend.enable_prefill_microbatch_overlap = False
    backend.enable_decode_microbatch_overlap = False

    backend.init_spec_engine()

    assert not hasattr(backend.spec_engine.planner, "_draft_step_group")


def test_dp_backend_overlap_decode_reuses_common_lightspec_group(monkeypatch):
    created_group = object()
    created_stream = object()
    real_torch_zeros = torch.zeros

    monkeypatch.setattr(lightspec_planner_impl, "get_global_world_size", lambda: 2)
    monkeypatch.setattr(lightspec_planner_impl.dist, "new_group", lambda **kwargs: created_group)
    monkeypatch.setattr(
        lightspec_planner_impl.torch,
        "zeros",
        lambda *args, **kwargs: real_torch_zeros(
            *args, **{key: value for key, value in kwargs.items() if key != "device"}
        ),
    )
    monkeypatch.setattr(lightspec_planner_impl.torch.cuda, "Stream", lambda: created_stream)

    backend = DPChunkedPrefillBackend.__new__(DPChunkedPrefillBackend)
    backend.args = SimpleNamespace(mtp_mode="eagle3", mtp_dynamic_verify=True, dp=2)
    backend.max_draft_step = 3
    backend.dp_size = 2
    backend.model = SimpleNamespace(graph=None)
    backend.draft_models = [SimpleNamespace(graph=None)]
    backend.enable_prefill_microbatch_overlap = False
    backend.enable_decode_microbatch_overlap = True

    backend.init_spec_engine()

    assert backend.spec_engine.planner._draft_step_group is created_group
    assert backend.spec_engine.planner._draft_step_stream is created_stream
    assert backend.decode_draft_engine is backend.dp_overlap_spec_engine
    assert backend.decode_draft_engine.common_engine.planner is backend.spec_engine.planner


def test_dp_backend_keeps_dynamic_planning_before_global_reduction():
    backend = DPChunkedPrefillBackend.__new__(DPChunkedPrefillBackend)
    backend.args = SimpleNamespace(mtp_mode="eagle3", mtp_dynamic_verify=True, dp=1)
    backend.max_draft_step = 3
    backend.dp_size = 1
    backend.model = SimpleNamespace(graph=None)
    backend.draft_models = [SimpleNamespace(graph=None)]
    backend.enable_prefill_microbatch_overlap = False
    backend.enable_decode_microbatch_overlap = False

    backend.init_spec_engine()

    assert isinstance(backend.spec_engine.planner, LightSpecPlanner)
    assert backend.spec_engine.proposer.enable_dynmaic_mtp
    assert backend.spec_engine.planner._draft_step_group is None


def test_dp_prefill_and_decode_select_overlap_engine_independently():
    args = SimpleNamespace(mtp_mode="eagle3", mtp_dynamic_verify=False, dp=1)

    for prefill_overlap in (False, True):
        for decode_overlap in (False, True):
            backend = DPChunkedPrefillBackend.__new__(DPChunkedPrefillBackend)
            backend.args = args
            backend.max_draft_step = 2
            backend.dp_size = 1
            backend.enable_prefill_microbatch_overlap = prefill_overlap
            backend.enable_decode_microbatch_overlap = decode_overlap
            backend.init_spec_engine()

            expected_prefill_engine = backend.dp_overlap_spec_engine if prefill_overlap else backend.spec_engine
            expected_decode_engine = backend.dp_overlap_spec_engine if decode_overlap else backend.spec_engine
            assert backend.prefill_draft_engine is expected_prefill_engine
            assert backend.decode_draft_engine is expected_decode_engine


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

    assert calls == [
        "plan",
        "prepare",
        "propose",
        "post_wait",
        "forward_wait",
        "free",
        "pre_post",
    ]


def test_dp_overlap_engine_delegates_raw_verify_layout_to_proposer():
    calls = {}

    class _Proposer:
        def propose_next_overlap(self, **kwargs):
            calls.update(kwargs)
            return SpecProposal(token_ids=kwargs["target_next_token_ids0"].new_empty((3, 7)))

    engine = DPOverlapSpecEngine.__new__(DPOverlapSpecEngine)
    engine.proposer = _Proposer()
    model_input0 = SimpleNamespace(batch_size=8)
    model_input1 = SimpleNamespace(batch_size=16)
    model_output0 = SimpleNamespace()
    model_output1 = SimpleNamespace()
    target_next_token_ids0 = torch.arange(8, dtype=torch.int64)
    target_next_token_ids1 = torch.arange(8, 24, dtype=torch.int64)
    accept_len0 = torch.tensor([2], dtype=torch.int32)
    accept_len1 = torch.tensor([3, 4], dtype=torch.int32)

    proposal = engine.propose_next_overlap(
        target_model_input0=model_input0,
        target_model_output0=model_output0,
        target_next_token_ids0=target_next_token_ids0,
        accept_len0=accept_len0,
        target_model_input1=model_input1,
        target_model_output1=model_output1,
        target_next_token_ids1=target_next_token_ids1,
        accept_len1=accept_len1,
        draft_step=7,
    )

    assert calls["target_model_input0"] is model_input0
    assert calls["target_model_input1"] is model_input1
    assert calls["target_model_output0"] is model_output0
    assert calls["target_model_output1"] is model_output1
    assert calls["target_next_token_ids0"] is target_next_token_ids0
    assert calls["target_next_token_ids1"] is target_next_token_ids1
    assert calls["accept_len0"] is accept_len0
    assert calls["accept_len1"] is accept_len1
    assert calls["draft_step"] == 7
    assert proposal.token_ids.shape == (3, 7)


def test_dp_overlap_engine_distributes_dynamic_verify_budget_by_request_count():
    prepare_calls = []

    class _CommonEngine:
        def prepare_decode_model_input(self, **kwargs):
            prepare_calls.append(kwargs)
            model_input = kwargs["model_input"]
            model_input.batch_size = kwargs["plan"].dynamic_batch_size
            return model_input, f"mask{len(prepare_calls)}"

    engine = DPOverlapSpecEngine.__new__(DPOverlapSpecEngine)
    engine.common_engine = _CommonEngine()
    model_input0 = SimpleNamespace(batch_size=12)
    model_input1 = SimpleNamespace(batch_size=8)
    plan = SpecDecodePlan(
        origin_batch_size=20,
        dynamic_batch_size=11,
        draft_step=2,
        pre_draft_step=3,
    )

    compacted_input0, mask0, compacted_input1, mask1 = engine.prepare_decode_model_inputs(
        model_input0=model_input0,
        req_num0=3,
        model_input1=model_input1,
        req_num1=2,
        plan=plan,
    )

    assert compacted_input0.batch_size == 7
    assert compacted_input1.batch_size == 4
    assert mask0 == "mask1"
    assert mask1 == "mask2"
    assert prepare_calls[0]["req_num"] == 3
    assert prepare_calls[0]["plan"] == SpecDecodePlan(12, 7, 2, 3)
    assert prepare_calls[1]["req_num"] == 2
    assert prepare_calls[1]["plan"] == SpecDecodePlan(8, 4, 2, 3)


@pytest.mark.parametrize(
    ("batch_size0", "batch_size1", "dynamic_batch_size", "expected_batch_sizes"),
    (
        (4, 8, 9, (4, 5)),
        (8, 4, 10, (6, 4)),
    ),
)
def test_dp_overlap_engine_moves_verify_rows_to_the_side_with_capacity(
    batch_size0,
    batch_size1,
    dynamic_batch_size,
    expected_batch_sizes,
):
    dynamic_batch_sizes = []

    class _CommonEngine:
        def prepare_decode_model_input(self, **kwargs):
            dynamic_batch_sizes.append(kwargs["plan"].dynamic_batch_size)
            return kwargs["model_input"], None

    engine = DPOverlapSpecEngine.__new__(DPOverlapSpecEngine)
    engine.common_engine = _CommonEngine()
    engine.prepare_decode_model_inputs(
        model_input0=SimpleNamespace(batch_size=batch_size0),
        req_num0=2,
        model_input1=SimpleNamespace(batch_size=batch_size1),
        req_num1=2,
        plan=SpecDecodePlan(
            origin_batch_size=batch_size0 + batch_size1,
            dynamic_batch_size=dynamic_batch_size,
            draft_step=2,
            pre_draft_step=3,
        ),
    )

    assert tuple(dynamic_batch_sizes) == expected_batch_sizes


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_dp_overlap_decode_delegates_empty_layout_and_frees_proposal(monkeypatch):
    device = "cuda"
    empty_i32 = torch.empty((0,), dtype=torch.int32, device=device)
    model_input0 = SimpleNamespace(batch_size=0, b_req_idx=empty_i32, b_mtp_index=empty_i32)
    model_input1 = SimpleNamespace(batch_size=0, b_req_idx=empty_i32, b_mtp_index=empty_i32)
    model_output0 = SimpleNamespace(logits=torch.empty((0, 8), device=device))
    model_output1 = SimpleNamespace(logits=torch.empty((0, 8), device=device))
    calls = []

    class _OverlapEngine:
        def plan_decode(self, **kwargs):
            calls.append("plan")
            return SpecDecodePlan(0, 0, 2, 2)

        def prepare_decode_model_inputs(self, **kwargs):
            calls.append("prepare")
            return kwargs["model_input0"], None, kwargs["model_input1"], None

        def propose_next_overlap(self, **kwargs):
            calls.append("propose")
            assert kwargs["target_next_token_ids0"].shape == (0,)
            assert kwargs["target_next_token_ids0"].dtype == torch.int64
            assert kwargs["target_next_token_ids1"].shape == (0,)
            assert kwargs["target_next_token_ids1"].dtype == torch.int64
            assert kwargs["accept_len0"].shape == (0,)
            assert kwargs["accept_len0"].dtype == torch.int32
            assert kwargs["accept_len1"].shape == (0,)
            assert kwargs["accept_len1"].dtype == torch.int32
            assert kwargs["draft_step"] == 2
            return SpecProposal(token_ids=torch.empty((0, 2), dtype=torch.int64, device=device))

    backend = DPChunkedPrefillBackend.__new__(DPChunkedPrefillBackend)
    backend.decode_draft_engine = _OverlapEngine()
    backend.model = SimpleNamespace(
        microbatch_overlap_decode=lambda input0, input1: (model_output0, model_output1),
    )
    event_pack = SimpleNamespace(
        notify_post_handle_and_wait_pre_post_handle=lambda: calls.append("post_wait"),
        notify_forward_and_wait_post_handle=lambda: calls.append("forward_wait"),
        notify_pre_post_handle=lambda: calls.append("pre_post"),
    )
    monkeypatch.setattr(
        dp_backend_impl,
        "overlap_prepare_decode_inputs",
        lambda req_objs: (model_input0, [], [], model_input1, [], []),
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

    backend.decode_overlap_mtp(event_pack=event_pack, decode_reqs=[])

    assert calls == [
        "plan",
        "prepare",
        "propose",
        "post_wait",
        "forward_wait",
        "free",
        "pre_post",
    ]
