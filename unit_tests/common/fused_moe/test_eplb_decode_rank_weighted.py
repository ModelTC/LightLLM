"""加权 rank 矩阵路由：验证参考结果，以及与 hash_targets 共用的统计/Graph 约束。"""

import pytest
import torch
from unittest.mock import MagicMock

from lightllm.common.basemodel.triton_kernel.fused_moe import decode_eplb_topk_ids_rank_weighted as weighted_routing
from lightllm.common.basemodel.triton_kernel.fused_moe.decode_eplb_topk_ids_rank_weighted import (
    eplb_rank_weighted_topk_ids,
)
from lightllm.common.basemodel.triton_kernel.fused_moe.decode_eplb_topk_ids_hash_targets import (
    eplb_hash_targets_topk_ids,
)
from lightllm.server.router.model_infer.mode_backend.eplb.placement import (
    build_logical_to_rank_physical_map,
    build_initial_local_expert_ids,
    build_logical_to_physical_map,
)


def weighted_reference(
    logical, logical_to_rank_physical_map_cpu, source_rank, node_world_size, normalize_replica_coverage=True
):
    """返回每个 top-k 槽允许选择的物理副本集合，不假设同分时的设备归约顺序。"""
    result = []
    for experts in logical:
        remote = [expert for expert in experts if logical_to_rank_physical_map_cpu[expert][source_rank] < 0]
        weights = {
            expert: 1024 // sum(physical >= 0 for physical in logical_to_rank_physical_map_cpu[expert])
            if normalize_replica_coverage
            else 1
            for expert in remote
        }
        coverage = [
            sum(weights[expert] for expert in remote if logical_to_rank_physical_map_cpu[expert][rank] >= 0)
            for rank in range(len(logical_to_rank_physical_map_cpu[0]))
        ]
        scores = []
        for rank, count in enumerate(coverage):
            score = count
            score += 10000000 if rank // node_world_size == source_rank // node_world_size else 0
            scores.append(score)
        selected = []
        for expert in experts:
            local = logical_to_rank_physical_map_cpu[expert][source_rank]
            available = [
                rank for rank, physical in enumerate(logical_to_rank_physical_map_cpu[expert]) if physical >= 0
            ]
            best_score = max(scores[rank] for rank in available)
            selected.append(
                {local}
                if local >= 0
                else {
                    logical_to_rank_physical_map_cpu[expert][rank] for rank in available if scores[rank] == best_score
                }
            )
        result.append(selected)
    return result


def assert_weighted_matches_reference(
    output, logical, logical_to_rank_physical_map_cpu, source_rank, node_world_size, normalize_replica_coverage=True
):
    candidates = weighted_reference(
        logical, logical_to_rank_physical_map_cpu, source_rank, node_world_size, normalize_replica_coverage
    )
    assert tuple(output.shape) == (len(logical), len(logical[0]))
    for physical_row, candidate_row in zip(output.tolist(), candidates):
        assert all(physical in allowed for physical, allowed in zip(physical_row, candidate_row))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("normalize_replica_coverage", [False, True])
@pytest.mark.parametrize("tokens", [1, 32, 33, 64, 129])
@pytest.mark.parametrize("world,node,topk,redundant", [(1, 1, 6, 0), (3, 1, 8, 4), (4, 2, 6, 8), (8, 4, 33, 16)])
def test_weighted_matches_reference_and_preserves_logical_ids(
    dtype, normalize_replica_coverage, tokens, world, node, topk, redundant
):
    experts = world * 24 if world == 3 else 64
    placement = build_initial_local_expert_ids(experts, world, redundant)
    logical_to_rank_physical_map_cpu = build_logical_to_rank_physical_map(placement, experts)
    logical_cpu = (
        torch.rand((tokens, experts), generator=torch.Generator().manual_seed(20261008))
        .argsort(dim=1)[:, :topk]
        .to(dtype)
    )
    logical = logical_cpu.contiguous().cuda()
    physical_to_logical = torch.tensor(placement, dtype=torch.int64, device="cuda").flatten()
    for source_rank in (0, world - 1):
        kwargs = dict(
            logical_topk_ids=logical,
            logical_to_rank_physical_map=torch.tensor(
                logical_to_rank_physical_map_cpu, dtype=torch.int32, device="cuda"
            ),
            current_rank=source_rank,
            node_world_size=node,
            normalize_replica_coverage=normalize_replica_coverage,
        )
        output = eplb_rank_weighted_topk_ids(**kwargs)
        assert output.dtype == dtype
        assert_weighted_matches_reference(
            output,
            logical_cpu.tolist(),
            logical_to_rank_physical_map_cpu,
            source_rank,
            node,
            normalize_replica_coverage,
        )
        assert torch.equal(physical_to_logical[output.long()], logical.long())
        assert torch.equal(logical.cpu(), logical_cpu)
        kwargs["logical_topk_ids"] = logical.flip(1).contiguous()
        reordered_output = eplb_rank_weighted_topk_ids(**kwargs).flip(1)
        # top-k 重排后仍需选最高分副本，但同分副本不要求与重排前完全相同。
        assert_weighted_matches_reference(
            reordered_output,
            logical_cpu.tolist(),
            logical_to_rank_physical_map_cpu,
            source_rank,
            node,
            normalize_replica_coverage,
        )
        assert torch.equal(physical_to_logical[reordered_output.long()], logical.long())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("normalize_replica_coverage", [False, True])
def test_weighted_graph_updates_layout_and_recording_flag(normalize_replica_coverage):
    placement = [[0, 1, 4], [0, 1, 5], [0, 2, 7], [0, 3, 6]]
    updated = [[0, 1, 7], [0, 1, 4], [0, 2, 5], [0, 3, 6]]
    logical_to_rank_physical_map_cpu = build_logical_to_rank_physical_map(placement, 8)
    logical_to_rank_physical_map = torch.tensor(logical_to_rank_physical_map_cpu, dtype=torch.int32, device="cuda")
    logical = torch.tensor([[1, 5, 0, 7], [4, 2, 6, 3]], dtype=torch.int32, device="cuda")
    counter = torch.zeros((8, 8), dtype=torch.int64, device="cuda")
    enabled = torch.zeros((1,), dtype=torch.int32, device="cuda")
    kwargs = dict(
        logical_topk_ids=logical,
        logical_to_rank_physical_map=logical_to_rank_physical_map,
        current_rank=2,
        node_world_size=2,
        route_counter=counter,
        update_route_counter=True,
        route_recording_enabled=enabled,
        normalize_replica_coverage=normalize_replica_coverage,
    )
    eplb_rank_weighted_topk_ids(**kwargs)
    addresses = (logical_to_rank_physical_map.data_ptr(), counter.data_ptr(), enabled.data_ptr())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = eplb_rank_weighted_topk_ids(**kwargs)
    counter.fill_(7)
    selected = torch.nn.functional.one_hot(logical.cpu().long(), num_classes=8).sum(dim=1)
    increment = torch.triu(selected.T @ selected)
    expected = torch.full((8, 8), 7, dtype=torch.int64)
    for iteration, flag in enumerate((0, 1, 0, 1, 1, 0)):
        logical_to_rank_physical_map_cpu = build_logical_to_rank_physical_map(
            updated if iteration % 2 else placement, 8
        )
        logical_to_rank_physical_map.copy_(
            torch.tensor(logical_to_rank_physical_map_cpu, dtype=torch.int32, device="cuda")
        )
        enabled.fill_(flag)
        graph.replay()
        assert_weighted_matches_reference(
            output, logical.tolist(), logical_to_rank_physical_map_cpu, 2, 2, normalize_replica_coverage
        )
        if not flag:
            snapshot = output.clone()
            graph.replay()
            assert torch.equal(output, snapshot)
        if flag:
            expected += increment
        assert torch.equal(counter.cpu(), expected)
        assert (logical_to_rank_physical_map.data_ptr(), counter.data_ptr(), enabled.data_ptr()) == addresses


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("shape", [(0, 8), (4, 0), (0, 0)])
def test_weighted_rejects_empty_topk_inputs(dtype, shape):
    # 空输入在分配输出及启动 kernel 之前失败，不用依赖 GPU 或特殊返回分支。
    with pytest.raises(AssertionError, match="logical_topk_ids must be non-empty"):
        eplb_rank_weighted_topk_ids(
            logical_topk_ids=torch.empty(shape, dtype=dtype),
            logical_to_rank_physical_map=torch.zeros((8, 4), dtype=torch.int32),
            current_rank=0,
            node_world_size=2,
        )


@pytest.mark.parametrize("parameter", ["block_tokens", "num_warps"])
def test_weighted_keeps_launch_parameters_internal(parameter):
    # 执行布局属于内部实现，不允许调用方覆盖。
    with pytest.raises(TypeError, match=parameter):
        eplb_rank_weighted_topk_ids(
            logical_topk_ids=torch.zeros((1, 1), dtype=torch.int32),
            logical_to_rank_physical_map=torch.zeros((1, 1), dtype=torch.int32),
            current_rank=0,
            node_world_size=1,
            **{parameter: 1},
        )


@pytest.mark.parametrize(
    "world,topk,tokens,block_tokens,num_warps",
    [
        (8, 8, 511, 1, 1),
        (8, 8, 512, 2, 1),
        (16, 8, 1024, 1, 1),
        (16, 8, 1535, 1, 1),
        (16, 8, 1536, 2, 1),
        (16, 8, 2047, 2, 1),
        (16, 8, 2048, 2, 1),
        (16, 8, 8191, 2, 1),
        (16, 8, 8192, 4, 1),
        (32, 8, 1023, 1, 4),
        (32, 8, 1024, 1, 1),
        (32, 8, 2047, 1, 1),
        (32, 8, 2048, 2, 1),
        (8, 16, 4096, 1, 1),
        (8, 33, 4096, 1, 4),
        (64, 8, 4096, 1, 4),
        (4, 8, 4096, 1, 1),
        (5, 8, 4096, 1, 1),
    ],
)
@pytest.mark.parametrize("normalize_replica_coverage", [False, True])
@pytest.mark.parametrize("update_route_counter", [False, True])
def test_weighted_selects_internal_layout_by_shape(
    monkeypatch, world, topk, tokens, block_tokens, num_warps, normalize_replica_coverage, update_route_counter
):
    # 只拦截 kernel 启动，直接验证内部选择及边界，不依赖 GPU 或重新开放调优参数。
    kernel = MagicMock()
    monkeypatch.setattr(weighted_routing, "_eplb_rank_weighted_matrix_kernel", kernel)
    output = eplb_rank_weighted_topk_ids(
        logical_topk_ids=torch.zeros((tokens, topk), dtype=torch.int32),
        logical_to_rank_physical_map=torch.zeros((64, world), dtype=torch.int32),
        current_rank=0,
        node_world_size=world,
        normalize_replica_coverage=normalize_replica_coverage,
        update_route_counter=update_route_counter,
        route_counter=torch.zeros((64, 64), dtype=torch.int64) if update_route_counter else None,
        route_recording_enabled=torch.ones((1,), dtype=torch.int32) if update_route_counter else None,
    )
    # 含统计的超大 batch 不能套用纯映射的四行布局。
    if world == 16 and topk == 8 and tokens >= 8192 and update_route_counter:
        block_tokens = 2
    assert output.shape == (tokens, topk)
    kernel.__getitem__.assert_called_once_with(((tokens + block_tokens - 1) // block_tokens,))
    launch_kwargs = kernel.__getitem__.return_value.call_args.kwargs
    assert launch_kwargs["BLOCK_TOKENS"] == block_tokens
    assert launch_kwargs["num_warps"] == num_warps


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    "world,tokens",
    [
        (8, 511),
        (8, 512),
        (16, 1535),
        (16, 1536),
        (16, 2047),
        (16, 2048),
        (16, 8193),
        (32, 1023),
        (32, 1024),
        (32, 2047),
        (32, 2048),
        (32, 4097),
    ],
)
@pytest.mark.parametrize("normalize_replica_coverage", [False, True])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_weighted_large_batch_graph_preserves_routing_and_statistics(world, tokens, normalize_replica_coverage, dtype):
    # 覆盖分块/warp 切换前后、奇数尾行与两种评分；统计必须精确且不能受分块变化影响。
    placement = build_initial_local_expert_ids(128, world, 4)
    logical_to_rank_physical_map_cpu = build_logical_to_rank_physical_map(placement, 128)
    logical_cpu = (
        torch.rand((tokens, 128), generator=torch.Generator().manual_seed(20261009)).topk(k=8, dim=1).indices.to(dtype)
    )
    logical = logical_cpu.contiguous().cuda()
    counter = torch.zeros((128, 128), dtype=torch.int64, device="cuda")
    kwargs = dict(
        logical_topk_ids=logical,
        logical_to_rank_physical_map=torch.tensor(logical_to_rank_physical_map_cpu, dtype=torch.int32, device="cuda"),
        current_rank=0,
        node_world_size=min(world, 8),
        normalize_replica_coverage=normalize_replica_coverage,
        route_counter=counter,
        update_route_counter=True,
        route_recording_enabled=torch.ones((1,), dtype=torch.int32, device="cuda"),
    )
    eplb_rank_weighted_topk_ids(**kwargs)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = eplb_rank_weighted_topk_ids(**kwargs)
    counter.zero_()
    graph.replay()
    assert output.dtype == dtype
    assert_weighted_matches_reference(
        output, logical_cpu.tolist(), logical_to_rank_physical_map_cpu, 0, min(world, 8), normalize_replica_coverage
    )
    physical_to_logical = torch.tensor(placement, dtype=torch.int64, device="cuda").flatten()
    assert torch.equal(physical_to_logical[output.long()], logical.long())
    selected = torch.nn.functional.one_hot(logical_cpu.long(), num_classes=128).sum(dim=1)
    increment = torch.triu(selected.T @ selected)
    assert torch.equal(counter.cpu(), increment)
    # 同一个大 batch Graph 内切换设备开关，布局不变；关闭时保留非零统计，而不是清零。
    counter.fill_(7)
    kwargs["route_recording_enabled"].zero_()
    graph.replay()
    assert bool((counter == 7).all())
    kwargs["route_recording_enabled"].fill_(1)
    graph.replay()
    assert torch.equal(counter.cpu(), increment + 7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("tokens", [8191, 8192, 8193])
@pytest.mark.parametrize("normalize_replica_coverage", [False, True])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_weighted_four_token_layout_preserves_experts_and_graph(tokens, normalize_replica_coverage, dtype):
    # W=16、无统计时覆盖四行分块边界及尾行，含统计的两行路径已在上述测试中验证。
    placement = build_initial_local_expert_ids(128, 16, 4)
    logical_to_rank_physical_map_cpu = build_logical_to_rank_physical_map(placement, 128)
    logical_cpu = (
        torch.rand((tokens, 128), generator=torch.Generator().manual_seed(2030)).topk(k=8, dim=1).indices.to(dtype)
    )
    logical = logical_cpu.contiguous().cuda()
    kwargs = dict(
        logical_topk_ids=logical,
        logical_to_rank_physical_map=torch.tensor(logical_to_rank_physical_map_cpu, dtype=torch.int32, device="cuda"),
        current_rank=15,
        node_world_size=8,
        normalize_replica_coverage=normalize_replica_coverage,
    )
    eplb_rank_weighted_topk_ids(**kwargs)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = eplb_rank_weighted_topk_ids(**kwargs)
    graph.replay()
    assert output.dtype == dtype
    assert_weighted_matches_reference(
        output, logical_cpu.tolist(), logical_to_rank_physical_map_cpu, 15, 8, normalize_replica_coverage
    )
    physical_to_logical = torch.tensor(placement, dtype=torch.int64, device="cuda").flatten()
    assert torch.equal(physical_to_logical[output.long()], logical.long())


@pytest.mark.parametrize("normalize_replica_coverage,topk", [(False, 10000000), (True, 9766)])
def test_weighted_rejects_topk_reaching_node_priority(normalize_replica_coverage, topk):
    # 覆盖数必须小于固定节点分，否则不能保证节点优先；断言在启动 GPU kernel 前执行。
    with pytest.raises(AssertionError):
        eplb_rank_weighted_topk_ids(
            logical_topk_ids=torch.zeros((1, topk), dtype=torch.int32),
            logical_to_rank_physical_map=torch.zeros((8, 4), dtype=torch.int32),
            current_rank=0,
            node_world_size=2,
            normalize_replica_coverage=normalize_replica_coverage,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("normalize_replica_coverage", [False, True])
def test_node_priority_overrides_remote_coverage_and_falls_back_globally(normalize_replica_coverage):
    # 本卡专家 0 在 rank 1 也有副本，但不能用它虚增 rank 1 的远端覆盖数。
    # 待分配的 1/2：rank 1 只含 1，rank 2 同时含 1/2；专家 3 只有 rank 3 能执行。
    placement = [[0, 4, 5], [0, 1, 6], [1, 2, 7], [3, 4, 5]]
    logical_to_rank_physical_map = torch.tensor(
        build_logical_to_rank_physical_map(placement, 8), dtype=torch.int32, device="cuda"
    )
    logical = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32, device="cuda")
    common = dict(
        logical_topk_ids=logical,
        logical_to_rank_physical_map=logical_to_rank_physical_map,
        current_rank=0,
        node_world_size=2,
        normalize_replica_coverage=normalize_replica_coverage,
    )
    output = eplb_rank_weighted_topk_ids(**common)
    # expert 1 的本节点副本优先，即使节点外覆盖数更大；2/3 无本节点副本，回退全局。
    assert output.tolist() == [[0, 4, 7, 9]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("normalize_replica_coverage", [False, True])
def test_local_experts_do_not_contribute_to_remote_coverage(normalize_replica_coverage):
    # expert 0 本卡命中，不能把其 rank 1 副本计入远端覆盖数。
    # 远端 expert 1 在 rank 1/2，expert 2 仅在 rank 2；应让 1/2 都选 rank 2。
    placement = [[0, 3, 4], [0, 1, 5], [1, 2, 6], [3, 4, 7]]
    output = eplb_rank_weighted_topk_ids(
        logical_topk_ids=torch.tensor([[0, 1, 2]], dtype=torch.int32, device="cuda"),
        logical_to_rank_physical_map=torch.tensor(
            build_logical_to_rank_physical_map(placement, 8), dtype=torch.int32, device="cuda"
        ),
        current_rank=0,
        node_world_size=4,
        normalize_replica_coverage=normalize_replica_coverage,
    )
    assert output.tolist() == [[0, 6, 7]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("normalize_replica_coverage", [False, True])
def test_tied_scores_select_one_replica_and_keep_local_experts(normalize_replica_coverage):
    # 远端 expert 1 在 rank 1/2 均有副本；两卡各覆盖 1 和另外一个远端专家，覆盖分相同。
    # 同分时可选 rank 1/2 的槽 4/7，但不能把两者相加；expert 0 始终保留本卡槽 9。
    placement = [[2, 3, 4], [0, 1, 5], [0, 1, 7], [0, 6, 2]]
    logical = torch.tensor([[1, 5, 7, 0]], dtype=torch.int32, device="cuda").expand(129, 4).contiguous()
    output = eplb_rank_weighted_topk_ids(
        logical_topk_ids=logical,
        logical_to_rank_physical_map=torch.tensor(
            build_logical_to_rank_physical_map(placement, 8), dtype=torch.int32, device="cuda"
        ),
        current_rank=3,
        node_world_size=4,
        normalize_replica_coverage=normalize_replica_coverage,
    )
    assert bool(((output[:, 0] == 4) | (output[:, 0] == 7)).all())
    expected_fixed = torch.tensor([[5, 8, 9]], dtype=logical.dtype, device="cuda").expand(129, 3)
    torch.testing.assert_close(output[:, 1:], expected_fixed, rtol=0, atol=0)
    physical_to_logical = torch.tensor(placement, dtype=torch.int64, device="cuda").flatten()
    torch.testing.assert_close(physical_to_logical[output.long()], logical.long(), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "routing_function,normalize_replica_coverage",
    [
        (eplb_hash_targets_topk_ids, False),
        (eplb_rank_weighted_topk_ids, False),
        (eplb_rank_weighted_topk_ids, True),
    ],
)
def test_decode_routing_preserves_weighted_expert_outputs(dtype, routing_function, normalize_replica_coverage):
    placement = build_initial_local_expert_ids(num_logical_experts=64, num_ranks=4, num_redundant_experts_per_rank=32)
    logical_to_rank_physical_map_cpu = build_logical_to_rank_physical_map(placement, 64)
    logical_ids = torch.rand((33, 64), device="cuda").topk(k=6, dim=1).indices.to(torch.int32)
    kwargs = dict(
        logical_topk_ids=logical_ids,
        logical_to_rank_physical_map=torch.tensor(logical_to_rank_physical_map_cpu, dtype=torch.int32, device="cuda"),
        current_rank=3,
        node_world_size=4,
    )
    if routing_function is eplb_hash_targets_topk_ids:
        candidates = build_logical_to_physical_map(placement, 64, current_rank=3, node_world_size=4)
        kwargs["logical_to_physical_map"] = torch.tensor(candidates, dtype=torch.int32, device="cuda")
        kwargs["num_physical_experts_per_rank"] = len(placement[0])
    else:
        kwargs["normalize_replica_coverage"] = normalize_replica_coverage
    physical_ids = routing_function(**kwargs)
    # 同一 logical 专家的物理副本使用完全相同的 SwiGLU 权重，
    # 验证逐槽专家输出及带 top-k 权重的最终输出，而不只验证 ID 范围。
    inputs = torch.randn((33, 1, 1, 8), dtype=dtype, device="cuda")
    w13 = torch.randn((64, 8, 16), dtype=dtype, device="cuda")
    w2 = torch.randn((64, 8, 8), dtype=dtype, device="cuda")
    routing_weights = torch.rand((33, 6, 1), dtype=dtype, device="cuda")
    snapshot = routing_weights.clone()
    physical_to_logical = torch.tensor(placement, device="cuda").flatten()

    def expert_output(ids, first_weights, second_weights):
        gate, up = torch.matmul(inputs, first_weights[ids.to(torch.long)]).squeeze(2).chunk(chunks=2, dim=-1)
        hidden = torch.nn.functional.silu(input=gate) * up
        return torch.matmul(hidden.unsqueeze(2), second_weights[ids.to(torch.long)]).squeeze(2)

    logical_output = expert_output(logical_ids, w13, w2)
    physical_output = expert_output(physical_ids, w13[physical_to_logical], w2[physical_to_logical])
    assert torch.equal(logical_output, physical_output)
    assert torch.equal((logical_output * routing_weights).sum(dim=1), (physical_output * routing_weights).sum(dim=1))
    assert torch.equal(snapshot, routing_weights)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("tokens", [1, 33, 64])
def test_inverse_replica_weights_prioritize_scarce_expert_ranks(tokens):
    # 0/1/2 各有三个副本，贡献 1024 // 3 = 341；3 只有一个副本，贡献 1024。
    # rank 1/3 覆盖 0/1/2，原模式分数 3；rank 2 覆盖 0/3，原模式分数 2。
    # 归一化后 rank 2 为 1365，超过 rank 1/3 的 1023，expert 0 应与稀缺 expert 3 共址。
    placement = [[4, 5, 6, 7, 8], [0, 1, 2, 4, 9], [0, 3, 5, 6, 7], [0, 1, 2, 8, 9], [1, 2, 4, 5, 9]]
    logical_to_rank_physical_map_cpu = build_logical_to_rank_physical_map(placement, 10)
    logical = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.int32, device="cuda").expand(tokens, 5).contiguous()
    common = dict(
        logical_topk_ids=logical,
        logical_to_rank_physical_map=torch.tensor(logical_to_rank_physical_map_cpu, dtype=torch.int32, device="cuda"),
        current_rank=0,
        node_world_size=5,
    )
    counted = eplb_rank_weighted_topk_ids(**common, normalize_replica_coverage=False)
    normalized = eplb_rank_weighted_topk_ids(**common, normalize_replica_coverage=True)
    # 不传开关时也应使用倒数加权，而不是原始覆盖数。
    assert torch.equal(eplb_rank_weighted_topk_ids(**common), normalized)
    assert bool(((counted[:, 0] == 5) | (counted[:, 0] == 15)).all())
    assert bool((normalized[:, 0] == 10).all())
    assert bool((normalized[:, 3] == 11).all())
    assert bool((normalized[:, 4] == 0).all())
    assert_weighted_matches_reference(normalized, logical.tolist(), logical_to_rank_physical_map_cpu, 0, 5, True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_normalized_graph_recomputes_weights_after_replica_count_changes():
    placement = [[4, 5, 6, 7, 8], [0, 1, 2, 4, 9], [0, 3, 5, 6, 7], [0, 1, 2, 8, 9], [1, 2, 4, 5, 9]]
    updated = [row.copy() for row in placement]
    updated[4][-1] = 3  # expert 3 从单副本变为双副本，权重从 1024 变为 512。
    logical_to_rank_physical_map = torch.tensor(
        build_logical_to_rank_physical_map(placement, 10), dtype=torch.int32, device="cuda"
    )
    logical = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.int32, device="cuda")
    kwargs = dict(
        logical_topk_ids=logical,
        logical_to_rank_physical_map=logical_to_rank_physical_map,
        current_rank=0,
        node_world_size=5,
        normalize_replica_coverage=True,
    )
    eplb_rank_weighted_topk_ids(**kwargs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = eplb_rank_weighted_topk_ids(**kwargs)
    for layout in (placement, updated, placement):
        logical_to_rank_physical_map_cpu = build_logical_to_rank_physical_map(layout, 10)
        logical_to_rank_physical_map.copy_(
            torch.tensor(logical_to_rank_physical_map_cpu, dtype=torch.int32, device="cuda")
        )
        graph.replay()
        assert_weighted_matches_reference(output, logical.tolist(), logical_to_rank_physical_map_cpu, 0, 5, True)
        assert output[0, 0].item() in ({5, 15} if layout is updated else {10})
