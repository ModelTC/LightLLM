import pytest
import torch

from lightllm.server.router.model_infer.mode_backend.eplb.placement import (
    DecodeCooccurrenceEPLBPlanner,
    build_logical_to_rank_physical_map,
    build_logical_to_physical_map,
    build_initial_local_expert_ids,
    create_eplb_planner,
)
from lightllm.server.router.model_infer.mode_backend.eplb.async_expert_transfer import build_transfer_plan


def _decode_statistics(num_ranks: int, layer_matrices: list[torch.Tensor]) -> torch.Tensor:
    """把单层对称测试矩阵转换为 planner 使用的 rank 分片上三角。"""
    upper_by_layer = torch.stack([torch.triu(matrix) for matrix in layer_matrices])
    statistics = torch.zeros(
        (
            num_ranks,
            len(layer_matrices),
            upper_by_layer.shape[-1],
            upper_by_layer.shape[-1],
        ),
        dtype=torch.int64,
    )
    # 故意把统计拆到两个源 rank，验证 planner 会先聚合 rank 维。
    statistics[0] = upper_by_layer // 2
    statistics[1] = upper_by_layer - statistics[0]
    return statistics


def _current_placement(num_experts: int, num_ranks: int, redundant: int, num_layers: int = 1):
    layer = build_initial_local_expert_ids(num_experts, num_ranks, redundant)
    return [[rank_experts[:] for rank_experts in layer] for _ in range(num_layers)]


@pytest.mark.parametrize(
    ("world_size", "redundant", "error"),
    [
        (0, 1, "world_size must be greater than one"),
        (1, 1, "world_size must be greater than one"),
        (2, 0, "num_redundant_experts_per_rank must be positive"),
        (2, -1, "num_redundant_experts_per_rank must be positive"),
    ],
)
def test_decode_planner_asserts_invalid_constructor_arguments(world_size, redundant, error):
    with pytest.raises(AssertionError, match=error):
        DecodeCooccurrenceEPLBPlanner(world_size=world_size, num_redundant_experts_per_rank=redundant)


@pytest.mark.parametrize(
    ("shape", "device", "error"),
    [
        ((2, 1, 4, 4), "meta", "must be on CPU"),
        ((2, 4, 4), "cpu", "must have shape"),
        ((1, 1, 4, 4), "cpu", "rank dimension does not match"),
        ((2, 0, 4, 4), "cpu", "square non-empty expert matrices"),
        ((2, 1, 0, 0), "cpu", "square non-empty expert matrices"),
        ((2, 1, 4, 3), "cpu", "square non-empty expert matrices"),
        ((2, 1, 5, 5), "cpu", "divisible by world_size"),
    ],
)
def test_decode_planner_asserts_invalid_statistics(shape, device, error):
    planner = DecodeCooccurrenceEPLBPlanner(world_size=2, num_redundant_experts_per_rank=1)
    statistics = torch.empty(shape, dtype=torch.int64, device=device)

    with pytest.raises(AssertionError, match=error):
        planner.plan(route_statistics=statistics, current_placement=[])


@pytest.mark.parametrize(
    ("placement", "error"),
    [
        ([], "layer count does not match"),
        ([[[0, 1, 2]]], "rank count does not match"),
        ([[[0, 1], [2, 3, 0]]], "invalid number of local experts"),
        ([[[0, 1, 1], [2, 3, 0]]], "duplicate experts on one rank"),
        ([[[0, 1, 4], [2, 3, 0]]], "does not cover every logical expert"),
    ],
)
def test_decode_planner_asserts_invalid_placement(placement, error):
    planner = DecodeCooccurrenceEPLBPlanner(world_size=2, num_redundant_experts_per_rank=1)
    statistics = torch.zeros((2, 1, 4, 4), dtype=torch.int64)

    with pytest.raises(AssertionError, match=error):
        planner.plan(route_statistics=statistics, current_placement=placement)


@pytest.mark.parametrize("num_layers", [1, 2])
def test_decode_planner_zeros_scoring_diagonal_without_changing_statistics(monkeypatch, num_layers):
    planner = DecodeCooccurrenceEPLBPlanner(world_size=2, num_redundant_experts_per_rank=1)
    statistics = torch.arange(2 * num_layers * 4 * 4, dtype=torch.int64).reshape(2, num_layers, 4, 4)
    original_statistics = statistics.clone()
    current = _current_placement(4, 2, 1, num_layers=num_layers)
    plan_layer = planner._plan_layer
    scoring_matrices = []

    def capture_scoring_matrix(*, cooccurrence_counts, current_layer):
        scoring_matrices.append(cooccurrence_counts.clone())
        return plan_layer(cooccurrence_counts=cooccurrence_counts, current_layer=current_layer)

    monkeypatch.setattr(planner, "_plan_layer", capture_scoring_matrix)
    result = planner.plan(route_statistics=statistics, current_placement=current)

    # 严格上三角恢复为对称矩阵：非对角共现不变，所有层的主对角线均为零。
    upper_counts = torch.triu(original_statistics.sum(dim=0).to(torch.float64), diagonal=1)
    expected_counts = upper_counts + upper_counts.transpose(-1, -2)
    assert torch.equal(torch.stack(scoring_matrices), expected_counts)
    assert torch.equal(statistics, original_statistics)

    # 恢复旧评分矩阵的自身计数，确认清零对角线不改变最终专家布局。
    previous_counts = expected_counts.clone()
    previous_counts.diagonal(dim1=-2, dim2=-1).copy_(original_statistics.sum(dim=0).diagonal(dim1=-2, dim2=-1))
    expected_placement = [
        plan_layer(cooccurrence_counts=previous_counts[layer], current_layer=current[layer])
        for layer in range(num_layers)
    ]
    assert result == expected_placement


def test_decode_cooccurrence_planner_keeps_primaries_and_aggregates_source_ranks():
    planner = DecodeCooccurrenceEPLBPlanner(world_size=2, num_redundant_experts_per_rank=2)
    cooccurrence = torch.eye(6, dtype=torch.int64) * 1000
    cooccurrence[0, 4] = cooccurrence[4, 0] = 100
    cooccurrence[1, 5] = cooccurrence[5, 1] = 80
    cooccurrence[4, 5] = cooccurrence[5, 4] = 70
    current = _current_placement(6, 2, 2)

    result = planner.plan(
        route_statistics=_decode_statistics(2, [cooccurrence]),
        current_placement=current,
    )

    assert result[0][0][:3] == [0, 1, 2]
    assert set(result[0][0][3:]) == {4, 5}
    assert result[0][1][:3] == [3, 4, 5]
    assert set(result[0][1][3:]) == {0, 1}
    assert all(len(rank_experts) == len(set(rank_experts)) for rank_experts in result[0])


def test_decode_cooccurrence_planner_can_copy_one_expert_to_multiple_ranks():
    planner = DecodeCooccurrenceEPLBPlanner(world_size=4, num_redundant_experts_per_rank=1)
    cooccurrence = torch.eye(8, dtype=torch.int64) * 1000
    # Expert 7 与前三个 rank 的主专家都有很强共现，应分别复制过去；
    # rank 3 已有其主槽，不能再次把 expert 7 放入自己的冗余槽。
    for expert in range(6):
        cooccurrence[expert, 7] = cooccurrence[7, expert] = 100

    result = planner.plan(
        route_statistics=_decode_statistics(4, [cooccurrence]),
        current_placement=_current_placement(8, 4, 1),
    )

    redundant = [rank_experts[-1] for rank_experts in result[0]]
    assert redundant[:3] == [7, 7, 7]
    assert redundant[3] != 7
    assert all(len(experts) == len(set(experts)) for experts in result[0])


def test_decode_planner_includes_previous_replica_in_next_score():
    planner = DecodeCooccurrenceEPLBPlanner(world_size=2, num_redundant_experts_per_rank=2)
    cooccurrence = torch.zeros((6, 6), dtype=torch.float64)
    cooccurrence[3, 0] = cooccurrence[0, 3] = 10
    cooccurrence[4, 0] = cooccurrence[0, 4] = 9
    # 第一轮选 expert 3；第二轮加入它的贡献后，expert 5 应超过 expert 4。
    cooccurrence[3, 5] = cooccurrence[5, 3] = 1000

    selected = planner._select_redundant_experts(
        cooccurrence,
        primary_experts_by_rank=[[0, 1, 2], [3, 4, 5]],
    )

    assert selected[0] == [3, 5]


def test_decode_planner_expands_replica_cooccurrence_chain():
    planner = DecodeCooccurrenceEPLBPlanner(world_size=2, num_redundant_experts_per_rank=3)
    cooccurrence = torch.zeros((8, 8), dtype=torch.int64)
    cooccurrence[0, 4] = cooccurrence[4, 0] = 10
    cooccurrence[0, 5] = cooccurrence[5, 0] = 9
    cooccurrence[0, 6] = cooccurrence[6, 0] = 1
    cooccurrence[4, 6] = cooccurrence[6, 4] = 100
    cooccurrence[6, 7] = cooccurrence[7, 6] = 200

    selected = planner._select_redundant_experts(
        cooccurrence_counts=cooccurrence,
        primary_experts_by_rank=[[0, 1, 2, 3], [4, 5, 6, 7]],
    )
    # 新副本逐轮参与评分：先选 4，再由 4 带入 6，最后由 6 带入 7。
    assert selected[0] == [4, 6, 7]
    result = planner.plan(_decode_statistics(2, [cooccurrence]), _current_placement(8, 2, 3))
    assert set(result[0][0][4:]) == {4, 6, 7}


@pytest.mark.parametrize(
    ("world_size", "num_primary", "redundant"),
    [(2, 1, 1), (2, 4, 3), (4, 2, 1), (4, 2, 3), (4, 2, 6), (8, 4, 16)],
)
def test_decode_planner_incremental_scores_match_full_recomputation(world_size, num_primary, redundant):
    planner = DecodeCooccurrenceEPLBPlanner(world_size=world_size, num_redundant_experts_per_rank=redundant)
    num_experts = world_size * num_primary
    generator = torch.Generator().manual_seed(20261008)
    upper_counts = torch.triu(torch.randint(0, 50, (num_experts, num_experts), generator=generator), diagonal=1)
    cooccurrence = upper_counts + upper_counts.T
    original_counts = cooccurrence.clone()
    primaries = [list(range(rank * num_primary, (rank + 1) * num_primary)) for rank in range(world_size)]
    expected = []
    for primary_experts in primaries:
        # 参考实现每轮重新计算候选与全部已有专家的分数，不复用增量分数。
        placed = primary_experts[:]
        candidates = set(range(num_experts)) - set(placed)
        selected = []
        for _ in range(redundant):
            expert = min(
                candidates, key=lambda expert: (-sum(cooccurrence[expert, anchor].item() for anchor in placed), expert)
            )
            selected.append(expert)
            placed.append(expert)
            candidates.remove(expert)
        expected.append(selected)

    actual = planner._select_redundant_experts(cooccurrence_counts=cooccurrence, primary_experts_by_rank=primaries)
    assert actual == expected
    assert torch.equal(cooccurrence, original_counts)


@pytest.mark.parametrize("scale", [1, 7, 1000])
def test_decode_planner_selects_by_counts_and_breaks_exact_ties_by_expert_id(scale):
    planner = DecodeCooccurrenceEPLBPlanner(world_size=2, num_redundant_experts_per_rank=2)
    cooccurrence = torch.zeros((6, 6), dtype=torch.int64)
    # Expert 3、4 的计数分解不同，但总分同为 18；平局必须选择较小 ID。
    # Expert 5 总分为 19，应排在两者前面。整体缩放不应改变排序。
    cooccurrence[3, :3] = torch.tensor([5, 6, 7])
    cooccurrence[4, :3] = torch.tensor([3, 7, 8])
    cooccurrence[5, :3] = torch.tensor([9, 9, 1])
    cooccurrence = (cooccurrence + cooccurrence.T) * scale

    selected = planner._select_redundant_experts(
        cooccurrence_counts=cooccurrence,
        primary_experts_by_rank=[[0, 1, 2], [3, 4, 5]],
    )
    assert selected[0] == [5, 3]

    statistics = _decode_statistics(2, [cooccurrence])
    current = _current_placement(6, 2, 2)
    result = planner.plan(route_statistics=statistics, current_placement=current)
    assert result[0][0] == [0, 1, 2, 3, 5]
    assert planner.plan(route_statistics=statistics, current_placement=result) == result


@pytest.mark.parametrize("redundant", [3, 4, 6])
def test_decode_planner_allows_more_redundant_slots_than_primary_slots(redundant):
    # E=8、W=4，每卡主专家只有两个；R 可达到六个非本地主专家。
    planner = DecodeCooccurrenceEPLBPlanner(world_size=4, num_redundant_experts_per_rank=redundant)
    cooccurrence = torch.eye(8, dtype=torch.int64) * 1000
    cooccurrence[:6, 7] = 100
    cooccurrence[7, :6] = 100
    statistics = _decode_statistics(4, [cooccurrence])
    current = _current_placement(8, 4, redundant)

    result = planner.plan(statistics, current)

    for rank, experts in enumerate(result[0]):
        assert experts[:2] == [rank * 2, rank * 2 + 1]
        assert len(experts) == 2 + redundant
        assert len(experts) == len(set(experts))
        if rank != 3:
            assert 7 in experts[2:]
        if redundant == 6:
            assert set(experts) == set(range(8))
    if redundant == 3:
        assert [set(experts[2:]) for experts in result[0]] == [{2, 3, 7}, {0, 1, 7}, {0, 1, 7}, {0, 1, 2}]
    # 验证现有传输协议能把同一个专家复制到多卡，且不会覆盖任何主槽。
    live = [experts[:] for experts in current[0]]
    batches = build_transfer_plan(current[0], result[0], 0, num_logical_experts=8, world_size=4)
    for batch in batches:
        received = [(transfer, live[transfer.source_rank][transfer.source_local_expert_index]) for transfer in batch]
        for transfer, expert in received:
            assert expert == transfer.source_logical_expert_id
            assert transfer.dest_local_expert_index >= 2
            live[transfer.dest_rank][transfer.dest_local_expert_index] = expert
        assert all(len(experts) == len(set(experts)) for experts in live)
    assert live == result[0]
    # 已选专家保留原槽，相同统计再次规划不会产生布局振荡。
    assert planner.plan(statistics, result) == result


def test_decode_planner_rejects_redundancy_beyond_unique_local_capacity():
    planner = DecodeCooccurrenceEPLBPlanner(world_size=4, num_redundant_experts_per_rank=7)
    statistics = torch.zeros((4, 1, 8, 8), dtype=torch.int64)

    with pytest.raises(AssertionError, match="non-local logical experts"):
        planner.plan(statistics, current_placement=[])


@pytest.mark.parametrize(("world_size", "redundant"), [(2, 1), (4, 1), (4, 3)])
@pytest.mark.parametrize("num_layers", [1, 2])
@pytest.mark.parametrize("reassign_ranks", [False, True])
def test_decode_planner_rejects_changed_primary_slots(world_size, redundant, num_layers, reassign_ranks):
    num_experts = world_size * 2
    planner = DecodeCooccurrenceEPLBPlanner(world_size=world_size, num_redundant_experts_per_rank=redundant)
    canonical = _current_placement(num_experts, world_size, redundant, num_layers=num_layers)
    # 主专家集合相同但槽序改变，或跨 rank 重排，都不满足 decode 的规范主槽要求。
    current = [[list(reversed(row)) for row in layer] for layer in canonical]
    if reassign_ranks:
        current = [list(reversed(layer)) for layer in canonical]
    original = [[row[:] for row in layer] for layer in current]
    statistics = torch.zeros((world_size, num_layers, num_experts, num_experts), dtype=torch.int64)

    with pytest.raises(AssertionError, match="canonical primary slots"):
        planner.plan(route_statistics=statistics, current_placement=current)
    assert current == original


def test_decode_planner_rejects_missing_primary_experts():
    planner = DecodeCooccurrenceEPLBPlanner(world_size=2, num_redundant_experts_per_rank=1)
    # 全局覆盖完整仍不够：rank 1 必须保持自己的规范主槽 [2, 3]。
    current = [[[0, 1, 2], [0, 1, 3]]]
    statistics = torch.zeros((2, 1, 4, 4), dtype=torch.int64)

    with pytest.raises(AssertionError, match="canonical primary slots"):
        planner.plan(route_statistics=statistics, current_placement=current)
    assert current == [[[0, 1, 2], [0, 1, 3]]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for decode routing")
@pytest.mark.parametrize("redundant", [3, 6])
def test_decode_routing_accepts_planned_multi_rank_replicas_and_large_redundancy(redundant):
    from lightllm.common.basemodel.triton_kernel.fused_moe.decode_eplb_topk_ids_hash_targets import (
        eplb_hash_targets_topk_ids,
    )

    planner = DecodeCooccurrenceEPLBPlanner(world_size=4, num_redundant_experts_per_rank=redundant)
    cooccurrence = torch.eye(8, dtype=torch.int64) * 1000
    cooccurrence[:6, 7] = 100
    cooccurrence[7, :6] = 100
    current = _current_placement(8, 4, redundant)
    placement = planner.plan(_decode_statistics(4, [cooccurrence]), current)[0]
    logical_ids = torch.tensor([[0, 2, 4, 7], [1, 3, 5, 6]], dtype=torch.int32, device="cuda")
    slots_per_rank = 2 + redundant
    for source_rank in range(4):
        routing_rows = build_logical_to_physical_map(placement, 8, current_rank=source_rank, node_world_size=4)
        logical_to_rank_physical_map = build_logical_to_rank_physical_map(placement, 8)
        physical_ids = eplb_hash_targets_topk_ids(
            logical_topk_ids=logical_ids,
            logical_to_physical_map=torch.tensor(routing_rows, dtype=torch.int32, device="cuda"),
            logical_to_rank_physical_map=torch.tensor(logical_to_rank_physical_map, dtype=torch.int32, device="cuda"),
            current_rank=source_rank,
            num_physical_experts_per_rank=slots_per_rank,
        )
        for token, mapped in zip(logical_ids.tolist(), physical_ids.tolist()):
            for expert, physical_id in zip(token, mapped):
                rank, slot = divmod(physical_id, slots_per_rank)
                assert placement[rank][slot] == expert
                if expert in placement[source_rank]:
                    assert rank == source_rank
        if redundant == 6:
            assert torch.all(physical_ids // slots_per_rank == source_rank)


def test_decode_cooccurrence_planner_ignores_lower_triangle_and_plans_layers_independently():
    planner = DecodeCooccurrenceEPLBPlanner(world_size=2, num_redundant_experts_per_rank=1)
    first = torch.eye(4, dtype=torch.int64) * 100
    first[0, 3] = first[3, 0] = 50
    second = torch.eye(4, dtype=torch.int64) * 100
    second[0, 2] = second[2, 0] = 60
    statistics = _decode_statistics(2, [first, second])
    statistics[:, :, 3, 0] = 1_000_000  # 下三角不是 counter 的有效数据。
    current = _current_placement(4, 2, 1, num_layers=2)

    result = planner.plan(route_statistics=statistics, current_placement=current)

    assert result[0][0][-1] == 3
    assert result[1][0][-1] == 2


def test_decode_cooccurrence_planner_uses_smallest_expert_id_for_zero_score_ties():
    planner = DecodeCooccurrenceEPLBPlanner(world_size=2, num_redundant_experts_per_rank=2)
    statistics = torch.zeros((2, 1, 6, 6), dtype=torch.int64)
    current = _current_placement(6, 2, 2)

    result = planner.plan(route_statistics=statistics, current_placement=current)

    assert result == current


@pytest.mark.parametrize(
    ("current_redundant", "selected_redundant", "expected"),
    [
        ([2], [2], [2]),
        ([2], [3], [3]),
        ([4, 3], [5, 4], [4, 5]),
        ([2, 3, 4, 5], [5, 4, 3, 2], [2, 3, 4, 5]),
        ([2, 3, 4, 5], [9, 8, 7, 6], [9, 8, 7, 6]),
        ([2, 3, 4, 5], [8, 5, 9, 3], [8, 3, 9, 5]),
        ([2, 3, 4, 5], [9, 8, 2, 7], [2, 9, 8, 7]),
        ([2, 3, 4, 5, 6, 7], [12, 6, 11, 3, 10, 9], [12, 3, 11, 10, 6, 9]),
    ],
)
@pytest.mark.parametrize("num_primary", [1, 2, 8])
def test_decode_cooccurrence_planner_reuses_selected_experts_in_current_slots(
    current_redundant, selected_redundant, expected, num_primary
):
    planner = DecodeCooccurrenceEPLBPlanner(world_size=2, num_redundant_experts_per_rank=len(selected_redundant))
    # 传入完整新旧布局：共同专家复用绝对槽位，主专家始终保留原来的位置。
    primary_experts = list(range(num_primary))
    current_placement = primary_experts + [expert + num_primary for expert in current_redundant]
    selected_experts = primary_experts + [expert + num_primary for expert in selected_redundant]
    expected = primary_experts + [expert + num_primary for expert in expected]
    original_current = current_placement[:]
    original_selected = selected_experts[:]

    ordered = planner._reuse_current_slots(
        selected_experts=selected_experts,
        current_rank_placement=current_placement,
    )

    assert ordered == expected
    assert current_placement == original_current
    assert selected_experts == original_selected


@pytest.mark.parametrize(
    ("selected_experts", "current_placement", "expected"),
    [
        ([], [], []),
        ([0, 1], [0, 1], [0, 1]),
        ([0, 1, 2, 3, 8, 9], [0, 1, 2, 3, 7, 8], [0, 1, 2, 3, 9, 8]),
        ([0, 1, 2, 5, 4], [0, 1, 2, 4, 3], [0, 1, 2, 4, 5]),
    ],
)
def test_decode_slot_reuse_does_not_read_planner_configuration(selected_experts, current_placement, expected):
    # 不设置任何 planner 配置，重排仅依赖传入的新选专家和旧布局。
    planner = object.__new__(DecodeCooccurrenceEPLBPlanner)

    result = planner._reuse_current_slots(
        selected_experts=selected_experts,
        current_rank_placement=current_placement,
    )

    assert result == expected


@pytest.mark.parametrize(("selected_experts", "current_placement"), [([0, 1], [0]), ([0], [0, 1])])
def test_decode_slot_reuse_rejects_different_layout_sizes(selected_experts, current_placement):
    planner = object.__new__(DecodeCooccurrenceEPLBPlanner)

    with pytest.raises(AssertionError, match="layouts with equal sizes"):
        planner._reuse_current_slots(selected_experts=selected_experts, current_rank_placement=current_placement)


def test_factory_registers_decode_cooccurrence_for_decode_only():
    planner = create_eplb_planner(
        plan_mode="decode_cooccurrence",
        run_mode="decode",
        num_ranks=2,
        num_redundant_experts_per_rank=1,
        expert_alignment=128,
        node_world_size=2,
    )
    assert isinstance(planner, DecodeCooccurrenceEPLBPlanner)

    with pytest.raises(AssertionError, match="is for 'decode'"):
        create_eplb_planner(
            plan_mode="decode_cooccurrence",
            run_mode="prefill",
            num_ranks=2,
            num_redundant_experts_per_rank=1,
            expert_alignment=128,
            node_world_size=2,
        )


def test_decode_rank_map_preserves_all_replicas():
    logical_to_rank_physical_map = build_logical_to_rank_physical_map(
        [[0, 1, 3], [2, 3, 0], [4, 5, 0]],
        num_logical_experts=6,
    )

    assert logical_to_rank_physical_map[0] == [0, 5, 8]
    assert logical_to_rank_physical_map[3] == [2, 4, -1]


def test_decode_rank_map_accepts_moved_primaries_and_multiple_replicas():
    placement = [[0, 1, 4], [0, 1, 5], [0, 2, 7], [0, 3, 6]]
    logical_to_rank_physical_map = build_logical_to_rank_physical_map(placement, 8)

    assert logical_to_rank_physical_map[0] == [0, 3, 6, 9]
    assert logical_to_rank_physical_map[1] == [1, 4, -1, -1]
    for expert, physical_ids in enumerate(logical_to_rank_physical_map):
        for physical_id in physical_ids:
            if physical_id >= 0:
                assert placement[physical_id // 3][physical_id % 3] == expert


@pytest.mark.parametrize(
    "placement",
    [
        [[0, 0], [1, 2]],
        [[0, 1], [0, 1]],
    ],
)
def test_decode_rank_map_rejects_duplicate_or_missing_experts(placement):
    with pytest.raises(AssertionError):
        build_logical_to_rank_physical_map(placement, 4)
