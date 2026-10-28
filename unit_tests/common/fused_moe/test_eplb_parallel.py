"""EPLB routing and output consistency follow the TP/SP layout for every model."""

from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe import fused_moe_weight as weight_module
from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.eplb_placement import build_logical_to_physical_map
from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.impl import deepgemm_impl as deepgemm_module
from lightllm.common.quantization.quantize_method import WeightPack
from lightllm.server.core.objs.start_args_type import StartArgs
from lightllm.utils import device_utils, dist_utils, envs_utils


def _cpu_weight_factory(monkeypatch):
    # Exercise the real common initialization without allocating CUDA buffers.
    monkeypatch.setattr(device_utils, "is_sm100_gpu", lambda: False)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda tensor: tensor)
    original_zeros = torch.zeros

    def cpu_zeros(*args, **kwargs):
        if kwargs.get("device") == "cuda":
            kwargs["device"] = "cpu"
        return original_zeros(*args, **kwargs)

    monkeypatch.setattr(torch, "zeros", cpu_zeros)
    monkeypatch.setattr(weight_module, "get_moe_capture_callback", lambda *args: None)

    def make_weight(rank, tp, node_size, *, world=8, tpsp=False, eplb=True, draft=False):
        args = StartArgs(
            run_mode="prefill",
            tp=world,
            dp=world // tp,
            enable_ep_moe=True,
            enable_prefill_eplb=eplb,
            enable_tpsp_mix_mode=tpsp,
            eplb_num_redundant_experts_per_rank=2,
        )
        monkeypatch.setattr(envs_utils, "get_env_start_args", lambda: args)
        monkeypatch.setattr(dist_utils, "get_node_world_size", lambda: node_size)
        weight = weight_module.FusedMoeWeight.__new__(weight_module.FusedMoeWeight)
        weight.n_routed_experts = 2 * world
        weight.global_rank_, weight.global_world_size = rank, world
        weight.tp_rank_, weight.tp_world_size_ = rank % tp, tp
        weight.enable_ep_moe = True
        weight._init_expert_parallel_state({"is_mtp_draft_model": draft})
        weight.layer_num_ = 0
        num_local_experts = weight.expert_parallel_state.num_total_physical_experts // world
        weight.w13 = weight.w2 = WeightPack(weight=torch.empty((num_local_experts, 1, 1)))
        weight.e_score_correction_bias = weight.per_expert_scale = None
        weight.scoring_func = "softmax"
        weight.fuse_moe_impl = weight_module.create_fuse_moe_impl(
            n_routed_experts=weight.n_routed_experts,
            num_fused_shared_experts=0,
            routed_scaling_factor=1.0,
            quant_method=None,
            expert_parallel_state=weight.expert_parallel_state,
        )
        # Keep the real implementation and communication boundary; replace GPU kernels.
        weight.fuse_moe_impl._select_experts = lambda input_tensor, **kwargs: (
            None,
            torch.zeros((input_tensor.shape[0], 1), dtype=torch.long),
            None,
        )
        monkeypatch.setattr(
            deepgemm_module, "fused_experts", lambda hidden_states, **kwargs: hidden_states.clone() + 100 * rank
        )
        return weight

    return make_weight


@pytest.fixture
def make_weight(monkeypatch):
    return _cpu_weight_factory(monkeypatch)


@pytest.mark.parametrize("tpsp", [False, True])
@pytest.mark.parametrize("tp,node_size", [(1, 4), (2, 4), (4, 4), (4, 2), (8, 4), (2, 1)])
def test_common_initialization_preserves_node_local_replicas_and_tp_consistency(make_weight, tp, node_size, tpsp):
    states = []
    for rank in range(8):
        state = make_weight(rank, tp, node_size, tpsp=tpsp).expert_parallel_state.eplb
        leader = rank // tp * tp
        same_node = leader // node_size == (leader + tp - 1) // node_size
        replicated = tp > 1 and not tpsp
        expected_rotation = leader if replicated and same_node else rank
        assert state.replica_rotation_rank == expected_rotation
        assert state.tp_output_sync_src == (leader if replicated and not same_node else None)
        original_maps, original_counts = build_logical_to_physical_map(
            state.physical_to_logical, 16, source_rank=rank, node_world_size=node_size
        )
        # TP consistency changes ordering, preserving every eligible local replica.
        torch.testing.assert_close(state.logical_to_physical_map.sort(-1).values, original_maps.sort(-1).values)
        torch.testing.assert_close(state.logical_replica_count, original_counts)
        states.append(state)
    if not tpsp and 1 < tp <= node_size:
        for rank, state in enumerate(states):
            torch.testing.assert_close(state.logical_to_physical_map, states[rank // tp * tp].logical_to_physical_map)


@pytest.mark.parametrize("batch_size", [0, 3])
@pytest.mark.parametrize(
    "prefill,eplb,tp,node_size,tpsp,draft,synchronize",
    [
        (True, True, 4, 2, False, False, True),
        (True, True, 4, 8, False, False, False),
        (False, True, 4, 2, False, False, False),
        (None, True, 4, 2, False, False, False),
        (True, False, 4, 2, False, False, False),
        (True, True, 1, 2, False, False, False),
        (True, True, 4, 2, True, False, False),
        # Static draft layouts need the same cross-node TP output consistency.
        (True, True, 4, 2, False, True, True),
    ],
)
def test_ep_impl_synchronizes_only_replicated_eplb_prefill(
    monkeypatch, make_weight, batch_size, prefill, eplb, tp, node_size, tpsp, draft, synchronize
):
    weight = make_weight(6, tp, node_size, tpsp=tpsp, eplb=eplb, draft=draft)
    group = object()
    broadcast = Mock(side_effect=lambda output, **kwargs: output.fill_(4))
    monkeypatch.setattr(dist, "get_world_size", lambda group: tp)
    monkeypatch.setattr(dist, "broadcast", broadcast)
    inputs = torch.ones((batch_size, 4))
    output = weight.experts(
        inputs,
        router_logits=None,
        top_k=2,
        renormalize=True,
        use_grouped_topk=False,
        topk_group=0,
        num_expert_group=0,
        is_prefill=prefill,
        infer_state=SimpleNamespace(dist_group=group),
    )
    if synchronize:
        broadcast.assert_called_once_with(output, src=4, group=group, async_op=False)
    else:
        broadcast.assert_not_called()
    torch.testing.assert_close(output, torch.full_like(inputs, 4 if synchronize else 601))


@pytest.mark.parametrize("prefill", [True, False, None])
def test_eplb_sync_preserves_output_and_order(monkeypatch, make_weight, prefill):
    weight = make_weight(6, 4, 2)
    inputs = torch.ones((3, 4))
    output = inputs.clone()
    group = object()
    events = []

    def compute(**kwargs):
        events.append("compute")
        return output

    def broadcast(tensor, **kwargs):
        events.append("broadcast")
        assert tensor is output
        assert kwargs == {"src": 4, "group": group, "async_op": False}
        tensor.add_(10)

    monkeypatch.setattr(deepgemm_module, "fused_experts", compute)
    monkeypatch.setattr(dist, "get_world_size", lambda group: 4)
    monkeypatch.setattr(dist, "broadcast", broadcast)
    result = weight.fuse_moe_impl._fused_experts(
        inputs,
        weight.w13,
        weight.w2,
        torch.ones((3, 2)),
        torch.zeros((3, 2), dtype=torch.int32),
        is_prefill=prefill,
        dist_group=group,
    )
    assert result is output
    assert events == (["compute", "broadcast"] if prefill is True else ["compute"])
    torch.testing.assert_close(output, inputs + (10 if prefill is True else 0))


def test_failed_eplb_compute_does_not_broadcast(monkeypatch, make_weight):
    weight = make_weight(6, 4, 2)
    error = RuntimeError("expert computation failed")

    def fail(**kwargs):
        raise error

    broadcast = Mock()
    monkeypatch.setattr(deepgemm_module, "fused_experts", fail)
    monkeypatch.setattr(dist, "get_world_size", lambda group: 4)
    monkeypatch.setattr(dist, "broadcast", broadcast)
    with pytest.raises(RuntimeError) as caught:
        weight.fuse_moe_impl._fused_experts(
            torch.ones((3, 4)), weight.w13, weight.w2, torch.ones((3, 2)), torch.zeros((3, 2)), is_prefill=True
        )
    assert caught.value is error
    broadcast.assert_not_called()


def _tp_output_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=4, timeout=timedelta(seconds=30))
    try:
        groups = [dist.new_group(ranks) for ranks in ([0, 1], [2, 3])]
        group = groups[rank // 2]
        with pytest.MonkeyPatch.context() as monkeypatch:
            make_weight = _cpu_weight_factory(monkeypatch)
            for tpsp in (False, True):
                # Each two-rank TP group spans simulated single-rank nodes.
                weight = make_weight(rank, 2, 1, world=4, tpsp=tpsp)
                input_value = rank + 1 if tpsp else 1
                inputs = torch.full((3, 4), input_value, dtype=torch.float32)
                output = weight.experts(
                    inputs,
                    router_logits=None,
                    top_k=2,
                    renormalize=True,
                    use_grouped_topk=False,
                    topk_group=0,
                    num_expert_group=0,
                    is_prefill=True,
                    infer_state=SimpleNamespace(dist_group=group),
                )
                expected_rank = rank if tpsp else rank // 2 * 2
                torch.testing.assert_close(output, torch.full_like(inputs, input_value + 100 * expected_rank))
    finally:
        dist.destroy_process_group()


def test_real_tp_groups_synchronize_independently_and_preserve_sp_tokens(tmp_path):
    mp.spawn(_tp_output_worker, args=((tmp_path / "gloo-init").as_uri(),), nprocs=4, join=True)
