"""Startup prepares sampling without retaining warmup samples or specializing ring slots."""

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.expert_parallel_state import (
    create_expert_parallel_state,
)
from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.impl.deepgemm_impl import FuseMoeDeepGEMM
from lightllm.common.basemodel.triton_kernel.fused_moe import grouped_topk
from lightllm.server.core.objs.start_args_type import StartArgs
from lightllm.server.router.model_infer.mode_backend.eplb_manager import EPLBManager
from lightllm.utils import device_utils, dist_utils, envs_utils


@pytest.fixture
def runtime(monkeypatch):
    args = StartArgs(enable_ep_moe=True, enable_prefill_eplb=True, run_mode="prefill")
    monkeypatch.setattr(device_utils, "is_sm100_gpu", lambda: False)
    monkeypatch.setattr(dist_utils, "get_node_world_size", lambda: 4)
    monkeypatch.setattr(envs_utils, "get_env_start_args", lambda: args)
    monkeypatch.setattr(envs_utils, "get_prefill_eplb_step_interval", lambda: 20)
    monkeypatch.setattr(envs_utils, "get_prefill_eplb_steady_sample_steps", lambda: 4)
    return args


@pytest.mark.parametrize("draft", [False, True])
@pytest.mark.parametrize("run_mode", ["prefill", "normal"])
def test_initial_sampling_follows_model_role(monkeypatch, runtime, draft, run_mode):
    runtime.run_mode = run_mode
    monkeypatch.setattr(torch.Tensor, "cuda", lambda tensor: tensor)
    zeros = torch.zeros
    monkeypatch.setattr(torch, "zeros", lambda *args, **kwargs: zeros(*args, **{**kwargs, "device": "cpu"}))
    from lightllm.distributed import dist_group_manager

    monkeypatch.setattr(dist_group_manager, "ll_num_experts", 264, raising=False)
    state = create_expert_parallel_state(256, 4, 0, tp_rank=0, tp_world_size=1, is_mtp_draft_model=draft).eplb
    assert state.recording is (not draft)
    assert state.recorded_sample_count == 0
    assert state.next_sample_index() == 0
    assert state.recorded_sample_count == (0 if draft else 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("capture", [False, True])
def test_startup_sampling_covers_serving_slots_without_jit_misses(monkeypatch, runtime, grouped, dtype, capture):
    knobs = pytest.importorskip("triton.knobs", reason="Triton JIT observation hooks required")
    kernel = grouped_topk.grouped_topk_kernel
    # Isolate the in-process cache so earlier tests cannot hide missing startup variants.
    monkeypatch.setattr(kernel, "device_caches", defaultdict(kernel.create_binder))
    parallel = create_expert_parallel_state(256, 4, 0, tp_rank=0, tp_world_size=1)
    state = parallel.eplb
    assert state.recording
    impl = FuseMoeDeepGEMM(
        n_routed_experts=256,
        num_fused_shared_experts=0,
        routed_scaling_factor=1.0,
        quant_method=SimpleNamespace(method_name="none"),
        expert_parallel_state=parallel,
    )
    generator = torch.Generator(device="cuda").manual_seed(20261027)
    logits = {n: torch.randn((n, 256), generator=generator, device="cuda").to(dtype) for n in (32, 1)}
    hidden = {n: torch.zeros((n, 16), device="cuda", dtype=torch.bfloat16) for n in logits}
    bias = torch.zeros(256, device="cuda") if grouped else None

    def select(tokens, preserve=False, prefill=True):
        return impl._select_experts(
            input_tensor=hidden[tokens],
            router_logits=logits[tokens],
            correction_bias=bias,
            top_k=8,
            renormalize=True,
            use_grouped_topk=grouped,
            topk_group=4 if grouped else 1,
            num_expert_group=8 if grouped else 1,
            scoring_func="sigmoid" if grouped else "softmax",
            is_prefill=prefill,
            preserve_logical_ids=preserve,
        )

    # These lengths are already part of model startup warmup. No serving slots
    # or additional recording modes are manually prewarmed here. Capture buffers
    # are initialized after model warmup, so logical-ID capture starts disabled.
    select(32)
    select(1)
    assert state.recorded_sample_count == 2
    assert state.route_counter[..., 0].sum().item() == (32 + 1) * 8
    manager = EPLBManager.__new__(EPLBManager)
    manager._eplb_states = [state]
    manager._reset_recorded_samples()
    torch.cuda.synchronize()
    assert state.recorded_sample_count == 0
    assert torch.count_nonzero(state.route_counter).item() == 0

    misses, compilations = [], []

    def on_miss(**kwargs):
        misses.append(kwargs["key"])

    def on_compile(**kwargs):
        if not kwargs["cache_hit"]:
            compilations.append(kwargs["src"].name)

    monkeypatch.setattr(knobs.runtime, "jit_cache_hook", on_miss)
    monkeypatch.setattr(knobs.compilation, "listener", on_compile)
    # Normal mode without decode graphs starts with prefill-only warmup.
    # Neither the first decode nor a temporary sampling pause may add a key.
    for tokens in (1, 32):
        _, physical, _ = select(tokens, capture, prefill=False)
        logical = state.physical_to_logical.flatten().cuda()[physical]
        torch.testing.assert_close(physical, state.logical_to_physical_map[logical, 0].long(), rtol=0, atol=0)
    manager._set_recording(False)
    select(32, capture)
    assert state.recorded_sample_count == 0
    assert torch.count_nonzero(state.route_counter).item() == 0
    manager._set_recording(True)
    capacity = state.route_counter.shape[0]
    for step in range(2 * capacity + 1):
        tokens = 1 if step % 3 == 0 else 32
        manager.before_prefill()
        _, physical, logical = select(tokens, capture)
        expected_ids = state.physical_to_logical.flatten().cuda()[physical]
        expected = torch.bincount(expected_ids.flatten(), minlength=256)
        torch.testing.assert_close(state.route_counter[step % capacity, :, 0], expected, rtol=0, atol=0)
        if capture:
            torch.testing.assert_close(expected_ids, logical, rtol=0, atol=0)
        if step % 5 == 0:
            before = state.route_counter.clone()
            count = state.recorded_sample_count
            select(tokens, capture, prefill=False)
            assert state.recorded_sample_count == count
            torch.testing.assert_close(state.route_counter, before, rtol=0, atol=0)
    torch.cuda.synchronize()
    assert state.recorded_sample_count == 2 * capacity + 1
    assert misses == []
    assert compilations == []


def test_load_recording_requires_counter():
    with pytest.raises(ValueError, match="requires an expert counter"):
        grouped_topk.triton_grouped_topk_eplb(
            hidden_states=torch.empty((1, 0)),
            gating_output=torch.zeros((1, 128)),
            correction_bias=None,
            topk=1,
            renormalize=True,
            num_expert_group=1,
            topk_group=1,
            scoring_func="softmax",
            logical_to_physical_map=None,
            logical_replica_count=None,
            expert_counter=None,
            sample_index=0,
            record_load=True,
            use_grouped_topk=False,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("tokens", [1, 32])
@pytest.mark.parametrize("renormalize", [False, True])
def test_global_routing_preserves_distinct_logits_after_softmax_rounding(runtime, tokens, renormalize):
    parallel = create_expert_parallel_state(128, 4, 0, tp_rank=0, tp_world_size=1)
    impl = FuseMoeDeepGEMM(
        n_routed_experts=128,
        num_fused_shared_experts=0,
        routed_scaling_factor=1.0,
        quant_method=SimpleNamespace(method_name="none"),
        expert_parallel_state=parallel,
    )
    logits = torch.full((tokens, 128), -20.0, device="cuda")
    logits[:, 0] = 0.0
    logits[:, 1] = 1e-8
    rounded = logits.softmax(-1)
    assert torch.equal(rounded[:, 0], rounded[:, 1])
    weights, _, logical = impl._select_experts(
        input_tensor=torch.zeros((tokens, 16), device="cuda", dtype=torch.bfloat16),
        router_logits=logits,
        correction_bias=None,
        top_k=1,
        renormalize=renormalize,
        use_grouped_topk=False,
        topk_group=1,
        num_expert_group=1,
        scoring_func="softmax",
        is_prefill=True,
        preserve_logical_ids=True,
    )
    expected_ids = logits.argmax(-1, keepdim=True)
    torch.testing.assert_close(logical, expected_ids, rtol=0, atol=0)
    expected_weights = logits.double().softmax(-1).gather(1, expected_ids)
    if renormalize:
        expected_weights /= expected_weights.sum(-1, keepdim=True)
    torch.testing.assert_close(weights.double(), expected_weights, rtol=1e-6, atol=0)
