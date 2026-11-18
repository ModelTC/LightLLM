"""Route replay oracles independent of the planner's uniform-load approximation."""
import pytest
import torch

from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.eplb_placement import (
    _estimate_rank_load,
    build_logical_to_physical_map,
    build_initial_redundant_expert_ids,
    expand_redundant_placement,
    get_max_reachable_replicas,
)
from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.eplb_planner import plan_full_experts


def _hash(token, expert, copies):
    value = token ^ (((expert + 1) * 0x9E3779B9) & 0xFFFFFFFF)
    value = ((value ^ (value >> 16)) * 0x7FEB352D) & 0xFFFFFFFF
    value = ((value ^ (value >> 15)) * 0x846CA68B) & 0xFFFFFFFF
    return (value ^ (value >> 16)) % copies


def _samples(ids_by_sample, experts, world, node_size, tp_size=1):
    load = torch.zeros((len(ids_by_sample), 1, world // node_size, experts), dtype=torch.int64)
    hist = torch.zeros((*load.shape, world * (world + 1) // 2), dtype=torch.int64)
    for sample, sources in enumerate(ids_by_sample):
        for rank, ids in enumerate(sources):
            token = torch.arange(len(ids))[:, None].expand_as(ids).flatten()
            flat = ids.flatten()
            node = rank // node_size
            load[sample, 0, node] += torch.bincount(flat, minlength=experts)
            for copies in range(1, world + 1):
                bucket = torch.zeros_like(flat) if len(ids) == 1 else _hash(token, flat, copies)
                bucket = (bucket + rank // tp_size * tp_size) % copies
                start = copies * (copies - 1) // 2
                hist[sample, 0, node, :, start : start + copies] += torch.bincount(
                    flat * copies + bucket, minlength=experts * copies
                ).reshape(experts, copies)
    return load, hist


def _route_oracle(ids_by_sample, layout, node_size, tp_size=1):
    world, capacity = layout.shape
    experts = int(layout.max()) + 1
    results = []
    rows = layout.tolist()
    for sources in ids_by_sample:
        counts = torch.zeros((world, experts), dtype=torch.int64)
        for source, ids in enumerate(sources):
            for token, selected in enumerate(ids.tolist()):
                for expert in selected:
                    owners = [rank for rank in range(world) if expert in rows[rank]]
                    local = [rank for rank in owners if rank // node_size == source // node_size]
                    owners = local or owners
                    bucket = 0 if len(ids) == 1 else _hash(token, expert, len(owners))
                    destination = owners[(bucket + source // tp_size * tp_size) % len(owners)]
                    counts[destination, expert] += 1
        results.append(((counts + 127) // 128 * 128).sum(-1))
    return torch.stack(results)[:, None]


@pytest.mark.parametrize("node_size", [1, 2, 4])
@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_hash_estimator_matches_exact_source_local_route_replay(node_size, tp_size):
    layout = torch.tensor([[0, 1, 6, 7], [0, 2, 3, 4], [0, 3, 4, 5], [1, 2, 5, 6]])
    rng = torch.Generator().manual_seed(791)
    samples = [[torch.rand((tokens, 8), generator=rng).argsort(-1)[:, :2] for _ in range(4)] for tokens in [96, 1, 257]]
    load, hist = _samples(samples, 8, 4, node_size, tp_size)
    actual = _estimate_rank_load(load, layout[None], 128, node_size, replica_hash_load=hist)
    torch.testing.assert_close(actual, _route_oracle(samples, layout, node_size, tp_size).double(), rtol=0, atol=0)
    candidate = plan_full_experts(
        load, layout[None], expert_alignment=128, node_world_size=node_size, replica_hash_load=hist
    )
    candidate_load = _estimate_rank_load(load, candidate, 128, node_size, replica_hash_load=hist)
    torch.testing.assert_close(
        candidate_load, _route_oracle(samples, candidate[0], node_size, tp_size).double(), rtol=0, atol=0
    )
    assert candidate_load.max(-1).values.sum() <= actual.max(-1).values.sum()


def test_three_replicas_crossing_128_row_boundary_are_not_underestimated():
    layout = torch.cat((torch.arange(288).reshape(4, 72), torch.tensor([[72, 144], [0, 216], [0, 1], [2, 3]])), 1)
    samples = [[torch.zeros((96, 1), dtype=torch.int64) for _ in range(4)]]
    load, hist = _samples(samples, 288, 4, 4)
    actual = _estimate_rank_load(load, layout[None], 128, 4, replica_hash_load=hist)
    assert actual[0, 0].tolist() == [128, 256, 256, 0]


def test_hash_sampling_bound_uses_reachable_node_holdings_not_ep_world_size():
    for redundant, expected in [(0, 1), (2, 16)]:
        placement = expand_redundant_placement(build_initial_redundant_expert_ids(256, 64, redundant), 256)
        assert get_max_reachable_replicas(placement, 256, 8) == expected


def test_exact_estimator_accepts_only_the_reachable_hash_counts():
    layout = torch.arange(8).reshape(4, 2)
    samples = [[torch.zeros((96, 1), dtype=torch.int64) for _ in range(4)]]
    load, hist = _samples(samples, 8, 4, 4)
    actual = _estimate_rank_load(load, layout[None], 128, 4, replica_hash_load=hist[..., :1])
    assert actual[0, 0].tolist() == [384, 0, 0, 0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("tokens", [1, 96])
@pytest.mark.parametrize("source", range(4))
@pytest.mark.parametrize("tp_size", [1, 4])
def test_fused_topk_records_exact_hash_buckets_for_all_replica_counts(tokens, source, tp_size):
    from lightllm.common.basemodel.triton_kernel.fused_moe.grouped_topk import triton_grouped_topk_eplb

    layout = torch.tensor([[0, 1, 6, 7], [0, 2, 3, 4], [0, 3, 4, 5], [1, 2, 5, 6]])
    phase = source // tp_size * tp_size
    maps, copies = build_logical_to_physical_map(
        layout, 8, source_rank=source, node_world_size=4, replica_rotation_rank=phase
    )
    counter = torch.zeros((1, 8, 11), dtype=torch.int64, device="cuda")
    _, physical, logical = triton_grouped_topk_eplb(
        hidden_states=torch.zeros((tokens, 32), device="cuda"),
        gating_output=torch.randn((tokens, 8), device="cuda"),
        correction_bias=None,
        topk=2,
        renormalize=True,
        num_expert_group=1,
        topk_group=1,
        scoring_func="sigmoid",
        logical_to_physical_map=maps.cuda(),
        logical_replica_count=copies.cuda(),
        expert_counter=counter,
        sample_index=0,
        record_load=True,
        use_grouped_topk=True,
        return_logical_ids=True,
        group_score_used_topk_num=1,
        replica_rotation_rank=phase,
    )
    ids = logical.cpu()
    empty = torch.empty((0, 2), dtype=torch.int64)
    sources = [empty] * 4
    sources[source] = ids
    load, hist = _samples([sources], 8, 4, 4, tp_size)
    expected = torch.cat((load[0, 0, 0, :, None], hist[0, 0, 0]), -1)
    torch.testing.assert_close(counter[0].cpu(), expected, rtol=0, atol=0)
    expected_physical = []
    flat = layout.flatten().tolist()
    for token, selected in enumerate(ids.tolist()):
        row = []
        for expert in selected:
            owners = [p for p, value in enumerate(flat) if value == expert]
            bucket = 0 if tokens == 1 else _hash(token, expert, len(owners))
            row.append(owners[(bucket + phase) % len(owners)])
        expected_physical.append(row)
    assert physical.cpu().tolist() == expected_physical
