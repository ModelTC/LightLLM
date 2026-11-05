"""临时目标的伪随机初选参考、共址正确性、图更新与 logical 共现统计。"""

import pytest
import torch

from lightllm.common.basemodel.triton_kernel.fused_moe.decode_eplb_topk_ids_hash_targets import (
    eplb_hash_targets_topk_ids,
)
from lightllm.server.router.model_infer.mode_backend.eplb.placement import (
    build_logical_to_rank_physical_map,
    build_initial_local_expert_ids,
    build_logical_to_physical_map,
)
from unit_tests.common.fused_moe.test_eplb_decode_random import token_random_reference


def rank_priority_reference(source_rank, token_index, target_rank, world_size):
    """共同 rank 的 uint32 Philox 分数；offset 按模 2**32 回绕。"""
    mask = (1 << 32) - 1
    rank_bits = (world_size - 1).bit_length()
    offset = ((token_index << rank_bits) | target_rank) & mask
    return token_random_reference(source_rank=source_rank | (1 << 32), token_index=offset)


def routing_reference(logical_ids, candidates, logical_to_rank_physical_map_cpu, source_rank, slots, node_world_size=1):
    initial, output = [], []
    for token_index, token in enumerate(logical_ids):
        selected = []
        route_random = token_random_reference(source_rank=source_rank, token_index=token_index)
        for expert in token:
            local = logical_to_rank_physical_map_cpu[expert][source_rank]
            global_count, node_count = candidates[expert][:2]
            count = node_count if node_count > 0 else global_count
            index = route_random % count
            selected.append(local if local >= 0 else candidates[expert][index + 3])
        anchors = [physical // slots for physical in selected]
        result = []
        for expert in token:
            local = logical_to_rank_physical_map_cpu[expert][source_rank]
            # 初选槽作为保底，按 top-k 的锚点顺序遍历；同分保留当前副本。
            chosen = selected[len(result)]
            best_score = rank_priority_reference(
                source_rank, token_index, chosen // slots, len(logical_to_rank_physical_map_cpu[expert])
            )
            for rank in anchors:
                physical = logical_to_rank_physical_map_cpu[expert][rank]
                score = rank_priority_reference(
                    source_rank, token_index, rank, len(logical_to_rank_physical_map_cpu[expert])
                )
                if physical >= 0 and score < best_score:
                    chosen, best_score = physical, score
            result.append(local if local >= 0 else chosen)
        initial.append(selected)
        output.append(result)
    return initial, output


@pytest.mark.parametrize("source_rank", [0, 3, 7])
@pytest.mark.parametrize("token_index", [0, 1, 127, (1 << 32) - 1])
def test_rank_priorities_are_uint32_and_repeatable(source_rank, token_index):
    # 分数允许相同；验证 uint32 高位/offset 回绕，以及固定输入可复现。
    ranks = list(range(256)) + [1 << 16, 1 << 31, (1 << 31) + 1, (1 << 32) - 1]
    priorities = [
        rank_priority_reference(source_rank=source_rank, token_index=token_index, target_rank=rank, world_size=256)
        for rank in ranks
    ]
    assert all(0 <= priority < (1 << 32) for priority in priorities)
    assert priorities == [
        rank_priority_reference(source_rank=source_rank, token_index=token_index, target_rank=rank, world_size=256)
        for rank in ranks
    ]


@pytest.mark.parametrize("world_size,redundant", [(1, 0), (3, 0), (3, 1), (4, 4), (4, 12), (8, 2)])
@pytest.mark.parametrize("single_node", [False, True])
def test_prefill_candidates_and_rank_table_match_layout(world_size, redundant, single_node):
    experts = world_size * 4
    placement = build_initial_local_expert_ids(experts, world_size, redundant)
    snapshot = [row[:] for row in placement]
    logical_to_rank_physical_map_cpu = build_logical_to_rank_physical_map(placement, experts)
    slots = len(placement[0])
    for source_rank in range(world_size):
        candidates = build_logical_to_physical_map(
            placement, experts, current_rank=source_rank, node_world_size=world_size if single_node else 1
        )
        assert len(candidates) == len(logical_to_rank_physical_map_cpu) == experts
        for expert in range(experts):
            physical_ids = [rank * slots + row.index(expert) for rank, row in enumerate(placement) if expert in row]
            count = candidates[expert][0]
            assert sorted(candidates[expert][3 : 3 + count]) == physical_ids
            assert candidates[expert][3 + count :] == [-1] * (world_size * slots - count)
            assert len(candidates[expert]) == 3 + world_size * slots
            assert logical_to_rank_physical_map_cpu[expert] == [
                rank * slots + row.index(expert) if expert in row else -1 for rank, row in enumerate(placement)
            ]
            assert count == sum(physical >= 0 for physical in logical_to_rank_physical_map_cpu[expert])
            # 单 rank 节点的 node_count 就是本卡副本数，远端专家必为 0。
            expected_node_count = (
                count if single_node else int(logical_to_rank_physical_map_cpu[expert][source_rank] >= 0)
            )
            assert candidates[expert][1] == expected_node_count
    assert placement == snapshot


@pytest.mark.parametrize(
    "placement,experts",
    [([], 4), ([[0, 1], [2]], 4), ([[0, 0], [2, 3]], 4), ([[0, 1], [0, 1]], 4), ([[0, -1], [2, 3]], 4)],
)
def test_rank_table_builder_rejects_invalid_layout(placement, experts):
    with pytest.raises(AssertionError):
        build_logical_to_rank_physical_map(placement, experts)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("single_node", [False, True])
@pytest.mark.parametrize("tokens", [0, 1, 33, 129])
@pytest.mark.parametrize(
    "world_size,redundant,topk",
    [(1, 0, 6), (3, 0, 8), (3, 2, 6), (4, 1, 1), (4, 8, 4), (4, 12, 8), (8, 2, 6), (8, 28, 33)],
)
def test_hash_routing_matches_reference_and_keeps_local_experts(
    dtype, tokens, world_size, redundant, topk, single_node
):
    experts = max(64, world_size * 8)
    if experts % world_size:
        experts = world_size * 24
    placement = build_initial_local_expert_ids(experts, world_size, redundant)
    logical_to_rank_physical_map_cpu = build_logical_to_rank_physical_map(placement, experts)
    slots = len(placement[0])
    logical_cpu = (
        torch.rand((tokens, experts), generator=torch.Generator().manual_seed(2026)).argsort(dim=1)[:, :topk].to(dtype)
    )
    physical_to_logical = torch.tensor(placement, dtype=torch.int64, device="cuda").flatten()
    kwargs = dict(
        logical_to_rank_physical_map=torch.tensor(logical_to_rank_physical_map_cpu, dtype=torch.int32, device="cuda"),
        num_physical_experts_per_rank=slots,
    )
    for source_rank in range(world_size):
        candidates = build_logical_to_physical_map(
            placement, experts, current_rank=source_rank, node_world_size=world_size if single_node else 1
        )
        kwargs["logical_to_physical_map"] = torch.tensor(candidates, dtype=torch.int32, device="cuda")
        kwargs["node_world_size"] = world_size if single_node else 1
        initial, expected = routing_reference(
            logical_cpu.tolist(),
            candidates,
            logical_to_rank_physical_map_cpu,
            source_rank,
            slots,
            node_world_size=kwargs["node_world_size"],
        )
        logical = logical_cpu.contiguous().cuda()
        result = eplb_hash_targets_topk_ids(logical_topk_ids=logical, current_rank=source_rank, **kwargs)
        expected_tensor = torch.tensor(expected, dtype=dtype).reshape(tokens, topk)
        assert torch.equal(result.cpu(), expected_tensor)
        assert torch.equal(physical_to_logical[result.long()], logical.long())
        assert torch.equal(logical.cpu(), logical_cpu)
        assert kwargs["logical_to_physical_map"].tolist() == candidates
        reversed_result = eplb_hash_targets_topk_ids(
            logical_topk_ids=logical.flip(1).contiguous(), current_rank=source_rank, **kwargs
        )
        assert torch.equal(reversed_result.flip(1), result)
        for token, before, after in zip(logical_cpu.tolist(), initial, result.tolist()):
            assert {physical // slots for physical in after} - {source_rank} <= {
                physical // slots for physical in before
            } - {source_rank}
            for expert, physical in zip(token, after):
                if logical_to_rank_physical_map_cpu[expert][source_rank] >= 0:
                    assert physical == logical_to_rank_physical_map_cpu[expert][source_rank]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("topk", [2, 4])
def test_shared_token_random_selects_one_rank_for_matching_replica_rank_sets(dtype, topk):
    # 热专家 0..3 在 rank 0/1/2 均有副本，源 rank 3 没有。
    # 同 token 的所有热专家应选同一个 rank，而不是互换初选槽；跨 token 则覆盖所有远端 rank。
    placement = [
        [0, 1, 2, 3, 4, 5, 6, 7],
        [4, 5, 6, 7, 0, 1, 2, 3],
        [8, 9, 10, 11, 0, 1, 2, 3],
        [12, 13, 14, 15, 8, 9, 10, 11],
    ]
    source_rank, slots, tokens = 3, 8, 4096
    candidates = build_logical_to_physical_map(placement, 16, current_rank=source_rank, node_world_size=4)
    logical_to_rank_physical_map_cpu = build_logical_to_rank_physical_map(placement, 16)
    logical = torch.arange(topk, dtype=dtype, device="cuda").expand(tokens, topk).contiguous()
    kwargs = dict(
        logical_topk_ids=logical,
        logical_to_physical_map=torch.tensor(candidates, dtype=torch.int32, device="cuda"),
        logical_to_rank_physical_map=torch.tensor(logical_to_rank_physical_map_cpu, dtype=torch.int32, device="cuda"),
        current_rank=source_rank,
        num_physical_experts_per_rank=slots,
    )
    initial, expected = routing_reference(
        logical.tolist(), candidates, logical_to_rank_physical_map_cpu, source_rank, slots
    )
    result = eplb_hash_targets_topk_ids(**kwargs)
    ranks = result // slots
    assert torch.equal(result.cpu(), torch.tensor(expected, dtype=dtype))
    assert torch.all(ranks == ranks[:, :1])
    # 候选 rank 集合及其排序相同时，共享随机数让初选就落在同一个 rank。
    assert all(len({physical // slots for physical in row}) == 1 for row in initial)
    # 检查不同 token 的偏好确实分散到中间 rank；这不是任意布局的均衡保证。
    counts = torch.bincount(ranks[:, 0].long(), minlength=4).cpu()
    assert counts[source_rank] == 0
    assert torch.all((counts[:3] > tokens * 0.20) & (counts[:3] < tokens * 0.45))
    assert torch.equal(result, eplb_hash_targets_topk_ids(**kwargs))
    reversed_kwargs = dict(kwargs, logical_topk_ids=logical.flip(1).contiguous())
    assert torch.equal(eplb_hash_targets_topk_ids(**reversed_kwargs).flip(1), result)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("node_world_size", [1, 2, 4, 8])
@pytest.mark.parametrize("source_rank", [0, 3, 7])
def test_initial_sample_prefers_node_replicas_and_falls_back_to_global(dtype, node_world_size, source_rank):
    # 主槽每卡 4 个。热专家 0..3 在 rank 0..6 都有副本，rank 7 没有；
    # 其他专家提供本卡、节点内和仅全局可用的路径，四种节点大小均使用同一布局。
    placement = [
        list(range(rank * 4, rank * 4 + 4)) + (list(range(4, 8)) if rank in (0, 7) else list(range(4)))
        for rank in range(8)
    ]
    candidates = build_logical_to_physical_map(placement, 32, current_rank=source_rank, node_world_size=node_world_size)
    logical_to_rank_physical_map_cpu = build_logical_to_rank_physical_map(placement, 32)
    # K=1 时没有其他专家锚点，最终输出就是初选槽，可直接检查 sampled_physical_ids。
    logical = (torch.arange(4097, dtype=dtype, device="cuda") % 32).reshape(-1, 1)
    kwargs = dict(
        logical_topk_ids=logical,
        logical_to_physical_map=torch.tensor(candidates, dtype=torch.int32, device="cuda"),
        logical_to_rank_physical_map=torch.tensor(logical_to_rank_physical_map_cpu, dtype=torch.int32, device="cuda"),
        current_rank=source_rank,
        num_physical_experts_per_rank=8,
        node_world_size=node_world_size,
    )
    initial, expected = routing_reference(
        logical.tolist(), candidates, logical_to_rank_physical_map_cpu, source_rank, 8, node_world_size=node_world_size
    )
    result = eplb_hash_targets_topk_ids(**kwargs)
    assert result.tolist() == initial == expected
    for expert, physical in zip(logical.flatten().tolist(), result.flatten().tolist()):
        if logical_to_rank_physical_map_cpu[expert][source_rank] >= 0:
            assert physical == logical_to_rank_physical_map_cpu[expert][source_rank]
        elif candidates[expert][1] > 0:
            assert (physical // 8) // node_world_size == source_rank // node_world_size
        else:
            assert physical in candidates[expert][3 : 3 + candidates[expert][0]]
    assert torch.equal(result, eplb_hash_targets_topk_ids(**kwargs))
    assert torch.equal(logical.flatten(), torch.arange(4097, dtype=dtype, device="cuda") % 32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_node_preference_only_changes_initial_sample_not_coalescing():
    placement = [[0, 1, 2, 4], [2, 3, 0, 6], [4, 5, 0, 2], [6, 7, 4, 2]]
    candidates = build_logical_to_physical_map(placement, 8, current_rank=0, node_world_size=2)
    logical_to_rank_physical_map_cpu = build_logical_to_rank_physical_map(placement, 8)
    logical = torch.tensor([[6, 7]], dtype=torch.int32, device="cuda").expand(129, 2).contiguous()
    initial, expected = routing_reference(
        logical.tolist(), candidates, logical_to_rank_physical_map_cpu, 0, 4, node_world_size=2
    )
    result = eplb_hash_targets_topk_ids(
        logical_topk_ids=logical,
        logical_to_physical_map=torch.tensor(candidates, dtype=torch.int32, device="cuda"),
        logical_to_rank_physical_map=torch.tensor(logical_to_rank_physical_map_cpu, dtype=torch.int32, device="cuda"),
        current_rank=0,
        num_physical_experts_per_rank=4,
        node_world_size=2,
    )
    # expert 6 初选节点内 rank 1，expert 7 无节点内副本，只能去 rank 3。
    assert all(row[0] // 4 == 1 and row[1] // 4 == 3 for row in initial)
    assert result.tolist() == expected
    # 按用户指定保留现有共址规则：rank 3 分数更优时，expert 6 仍可共址到节点外。
    assert torch.any(result[:, 0] // 4 == 3)


@pytest.mark.parametrize("node_world_size", [0, -1, 3, 9])
def test_hash_routing_rejects_invalid_node_world_size(node_world_size):
    with pytest.raises(AssertionError):
        eplb_hash_targets_topk_ids(
            logical_topk_ids=torch.zeros((1, 1), dtype=torch.int32),
            logical_to_physical_map=torch.zeros((8, 19), dtype=torch.int32),
            logical_to_rank_physical_map=torch.zeros((8, 8), dtype=torch.int32),
            current_rank=0,
            num_physical_experts_per_rank=2,
            node_world_size=node_world_size,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_hash_fallback_spreads_repeated_hot_expert_across_remote_replicas():
    placement = [[0, 1], [0, 2], [0, 3], [2, 3]]
    candidates = build_logical_to_physical_map(placement, 4, current_rank=3, node_world_size=4)
    logical_to_rank_physical_map_cpu = build_logical_to_rank_physical_map(placement, 4)
    logical = torch.zeros((1024, 1), dtype=torch.int32, device="cuda")
    kwargs = dict(
        logical_topk_ids=logical,
        logical_to_physical_map=torch.tensor(candidates, dtype=torch.int32, device="cuda"),
        logical_to_rank_physical_map=torch.tensor(logical_to_rank_physical_map_cpu, dtype=torch.int32, device="cuda"),
        current_rank=3,
        num_physical_experts_per_rank=2,
    )
    result = eplb_hash_targets_topk_ids(**kwargs)
    counts = torch.bincount((result // 2).flatten().long(), minlength=4).cpu()
    assert counts[3] == 0
    assert torch.all((counts[:3] > 250) & (counts[:3] < 430))
    assert torch.equal(result, eplb_hash_targets_topk_ids(**kwargs))
    # 单个远端专家没有其他锚点；输出就是 token 级伪随机初选结果。
    expected = [candidates[0][3 + token_random_reference(3, token) % 3] for token in range(1024)]
    assert result.flatten().tolist() == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_graph_replays_updated_candidates_and_exact_logical_statistics():
    placement = [[0, 1, 4], [0, 1, 5], [0, 2, 7], [0, 3, 6]]
    updated = [[0, 1, 7], [0, 1, 4], [0, 2, 5], [0, 3, 6]]
    candidates = build_logical_to_physical_map(placement, 8, current_rank=2, node_world_size=2)
    logical_to_rank_physical_map_cpu = build_logical_to_rank_physical_map(placement, 8)
    candidate_map = torch.tensor(candidates, dtype=torch.int32, device="cuda")
    logical_to_rank_physical_map = torch.tensor(logical_to_rank_physical_map_cpu, dtype=torch.int32, device="cuda")
    addresses = candidate_map.data_ptr(), logical_to_rank_physical_map.data_ptr()
    logical = torch.tensor([[1, 5, 0, 7], [4, 2, 6, 3]], dtype=torch.int32, device="cuda")
    counter = torch.zeros((8, 8), dtype=torch.int64, device="cuda")
    enabled = torch.ones((1,), dtype=torch.int32, device="cuda")
    kwargs = dict(
        logical_topk_ids=logical,
        logical_to_physical_map=candidate_map,
        logical_to_rank_physical_map=logical_to_rank_physical_map,
        current_rank=2,
        num_physical_experts_per_rank=3,
        route_counter=counter,
        update_route_counter=True,
        route_recording_enabled=enabled,
        node_world_size=2,
    )
    eplb_hash_targets_topk_ids(**kwargs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = eplb_hash_targets_topk_ids(**kwargs)
    counter.zero_()
    graph.replay()
    selected = torch.nn.functional.one_hot(logical.cpu().long(), num_classes=8).sum(dim=1)
    expected_counter = torch.triu(selected.T @ selected)
    assert torch.equal(counter.cpu(), expected_counter)
    assert (
        output.tolist()
        == routing_reference(logical.tolist(), candidates, logical_to_rank_physical_map_cpu, 2, 3, node_world_size=2)[1]
    )
    candidates = build_logical_to_physical_map(updated, 8, current_rank=2, node_world_size=2)
    logical_to_rank_physical_map_cpu = build_logical_to_rank_physical_map(updated, 8)
    candidate_map.copy_(torch.tensor(candidates, dtype=torch.int32, device="cuda"))
    logical_to_rank_physical_map.copy_(torch.tensor(logical_to_rank_physical_map_cpu, dtype=torch.int32, device="cuda"))
    enabled.zero_()
    graph.replay()
    assert (
        output.tolist()
        == routing_reference(logical.tolist(), candidates, logical_to_rank_physical_map_cpu, 2, 3, node_world_size=2)[1]
    )
    assert torch.equal(counter.cpu(), expected_counter)
    assert (candidate_map.data_ptr(), logical_to_rank_physical_map.data_ptr()) == addresses
