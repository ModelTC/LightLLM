"""Prefill / decode 共用的最小副本映射表 CPU 构建测试。"""

from collections import Counter
from copy import deepcopy
from itertools import combinations, product
import random

import pytest

from lightllm.server.router.model_infer.mode_backend.eplb.placement import build_minimal_logical_to_physical_map


def test_minimal_mapping_selects_smallest_unassigned_ids_and_keeps_physical_slots():
    # rank 0 选择 0/1，rank 1 跳过已选的 1，再选择 2/3。
    # 逻辑 ID 的排序只影响选择顺序，不改变布局中的物理槽位。
    placement = [[2, 0, 1], [3, 1, 2]]
    original = deepcopy(placement)
    mapping = build_minimal_logical_to_physical_map(rank_to_logic_expert_ids=placement, num_logical_experts=4)

    assert mapping == [1, 2, 5, 3]
    assert placement == original
    assert build_minimal_logical_to_physical_map(rank_to_logic_expert_ids=placement, num_logical_experts=4) == mapping


@pytest.mark.parametrize("world_size,redundant", [(1, 0), (2, 0), (2, 1), (8, 0), (8, 1), (8, 4), (8, 40), (16, 4)])
@pytest.mark.parametrize("shuffle_slots", [False, True])
def test_minimal_mapping_has_one_owner_and_equal_rank_quotas(world_size, redundant, shuffle_slots):
    num_experts = 256
    primary_per_rank = num_experts // world_size
    slots_per_rank = primary_per_rank + redundant
    placement = []
    generator = random.Random(0)
    for rank in range(world_size):
        primary = list(range(rank * primary_per_rank, (rank + 1) * primary_per_rank))
        # 可包含同一专家在多个 rank 上的冗余，且冗余数量可以超过主专家数。
        candidates = [expert for expert in range(num_experts) if expert not in primary]
        generator.shuffle(candidates)
        rank_experts = primary + candidates[:redundant]
        if shuffle_slots:
            generator.shuffle(rank_experts)
        placement.append(rank_experts)
    original = deepcopy(placement)

    mapping = build_minimal_logical_to_physical_map(rank_to_logic_expert_ids=placement, num_logical_experts=num_experts)

    assert len(mapping) == num_experts and len(set(mapping)) == num_experts
    assert Counter(physical_id // slots_per_rank for physical_id in mapping) == {
        rank: primary_per_rank for rank in range(world_size)
    }
    for expert, physical_id in enumerate(mapping):
        rank, slot = divmod(physical_id, slots_per_rank)
        assert placement[rank][slot] == expert
        assert rank == expert // primary_per_rank
    assert placement == original


@pytest.mark.parametrize(
    "placement,num_experts",
    [
        ([], 4),
        ([[0, 1]], 0),
        ([[0, 1], [2, 3]], 5),
        ([[0, 1, 2], [2, 3]], 4),
        ([[0, 1, 1], [2, 3, 0]], 4),
        ([[0, 1, 4], [2, 3, 0]], 4),
        ([[0, 1, 2], [0, 1, 2]], 4),
    ],
)
def test_minimal_mapping_rejects_invalid_layouts(placement, num_experts):
    with pytest.raises(AssertionError):
        build_minimal_logical_to_physical_map(rank_to_logic_expert_ids=placement, num_logical_experts=num_experts)


def test_minimal_mapping_fills_missing_experts_without_changing_assigned_slots():
    # rank 0 已选的 0/1 不改配；2 只能补到 rank 0，3 只能补到 rank 1，允许最终为 3:1。
    placement = [[0, 1, 2], [0, 1, 3]]
    original = deepcopy(placement)
    mapping = build_minimal_logical_to_physical_map(rank_to_logic_expert_ids=placement, num_logical_experts=4)
    assert mapping == [0, 1, 2, 5]
    assert Counter(physical_id // 3 for physical_id in mapping) == {0: 3, 1: 1}
    assert placement == original
    assert build_minimal_logical_to_physical_map(placement, 4) == mapping


def test_missing_experts_choose_less_assigned_rank_and_update_counts():
    # rank 0 已选 0/1，rank 1 候选不足。补选 2 时 rank 2/3 同数量，先选 rank 2；
    # 随后的 3 必须看到计数变化，选尚未分配专家的 rank 3，不能再次选 rank 2。
    placement = [[0, 1, 4], [0, 1, 5], [2, 3, 6], [2, 3, 7]]
    mapping = build_minimal_logical_to_physical_map(placement, 8)
    assert mapping == [0, 1, 6, 10, 2, 5, 8, 11]
    assert Counter(physical_id // 3 for physical_id in mapping) == {0: 3, 1: 1, 2: 2, 3: 2}


def test_missing_expert_can_prefer_higher_rank_with_fewer_assigned_experts():
    # 专家 3 在 rank 0/2 均可执行；rank 0 已有两个 owner，rank 2 只有一个，应选 rank 2。
    placement = [[0, 1, 3], [0, 1, 4], [2, 3, 5]]
    assert build_minimal_logical_to_physical_map(placement, 6) == [0, 1, 6, 7, 5, 8]


@pytest.mark.parametrize(
    "placement,expected_capacity",
    [
        ([[0, 1, 2], [0, 1, 2], [3, 4, 5]], 3),
        ([[0, 1, 2], [0, 1, 2], [3, 4, 5], [5, 6, 7]], 3),
        ([[0, 1, 2, 3, 4, 5, 6]] * 3 + [[5, 6, 7, 8, 9, 10, 11]], 5),
    ],
)
def test_minimal_mapping_accepts_layouts_that_cannot_have_equal_rank_quotas(placement, expected_capacity):
    # 第一种布局的 3/4/5 只能在 rank 2 执行；第二种布局的后两卡必须承担 5 个专家。
    # 第三种布局的 7..11 只能在 rank 3 执行，直接补齐真实槽位即可，不再搜索容量。
    num_experts = max(expert for row in placement for expert in row) + 1
    original = deepcopy(placement)
    mapping = build_minimal_logical_to_physical_map(placement, num_experts)
    counts = Counter(physical_id // len(placement[0]) for physical_id in mapping)
    assert max(counts.values()) == expected_capacity
    assert len(mapping) == len(set(mapping)) == num_experts
    for expert, physical_id in enumerate(mapping):
        rank, slot = divmod(physical_id, len(placement[0]))
        assert placement[rank][slot] == expert
    assert placement == original
    assert build_minimal_logical_to_physical_map(placement, num_experts) == mapping


@pytest.mark.parametrize("world_size,slots_per_rank", [(2, 2), (2, 3), (4, 2)])
def test_small_layouts_always_cover_experts_with_one_real_physical_slot(world_size, slots_per_rank):
    # 穷举四专家的小布局，验证补选总能覆盖所有专家，且不修改输入或使用虚构槽位。
    num_experts = 4
    rank_layouts = list(combinations(range(num_experts), slots_per_rank))
    for rows in product(rank_layouts, repeat=world_size):
        if set(expert for row in rows for expert in row) != set(range(num_experts)):
            continue
        placement = [list(row) for row in rows]
        original = deepcopy(placement)
        mapping = build_minimal_logical_to_physical_map(placement, num_experts)
        assert len(mapping) == len(set(mapping)) == num_experts
        assert -1 not in mapping
        for expert, physical_id in enumerate(mapping):
            rank, slot = divmod(physical_id, slots_per_rank)
            assert placement[rank][slot] == expert
        assert placement == original
        assert build_minimal_logical_to_physical_map(placement, num_experts) == mapping


@pytest.mark.parametrize("world_size,redundant", [(2, 1), (8, 1), (8, 4), (8, 40), (16, 4)])
@pytest.mark.parametrize("seed", range(5))
def test_random_layouts_keep_one_real_slot_per_expert_without_reassigning_greedy_owners(world_size, redundant, seed):
    generator = random.Random(seed)
    num_experts = 256
    quota = num_experts // world_size
    experts = list(range(num_experts))
    generator.shuffle(experts)
    placement = []
    for rank in range(world_size):
        primary = experts[rank * quota : (rank + 1) * quota]
        candidates = [expert for expert in experts if expert not in primary]
        row = primary + generator.sample(candidates, redundant)
        generator.shuffle(row)
        placement.append(row)
    original = deepcopy(placement)
    mapping = build_minimal_logical_to_physical_map(placement, num_experts)
    assert len(mapping) == len(set(mapping)) == num_experts
    assert -1 not in mapping
    for expert, physical_id in enumerate(mapping):
        rank, slot = divmod(physical_id, quota + redundant)
        assert placement[rank][slot] == expert
    # 重新独立扫描初始贪心前缀，验证补选没有改动任何已选 owner。
    assigned_experts = set()
    for rank, row in enumerate(placement):
        candidates = sorted(set(row) - assigned_experts)
        if len(candidates) < quota:
            break
        for expert in candidates[:quota]:
            assert mapping[expert] == rank * (quota + redundant) + row.index(expert)
            assigned_experts.add(expert)
    assert placement == original
    assert build_minimal_logical_to_physical_map(placement, num_experts) == mapping
