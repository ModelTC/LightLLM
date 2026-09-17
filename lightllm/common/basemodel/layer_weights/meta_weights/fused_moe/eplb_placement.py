from typing import Dict, Tuple
import torch


def build_initial_local_expert_ids(
    num_logical_experts: int,
    num_ranks: int,
    num_redundant_experts_per_rank: int,
) -> list[list[int]]:
    """构建每个 rank 初始持有的完整 logical expert ID 列表。

    每个 rank 先持有连续划分得到的主专家，再按 rank 顺序选择不属于
    本 rank 的专家作为默认冗余副本。这里仅负责生成 Python 列表；调用方如果要参与 tensor
    运算，需要自行转换为 ``torch.Tensor``。

    例如 ``num_logical_experts=8``、``num_ranks=4``、每个 rank 有 2 个
    额外槽时，每个 rank 分到 2 个主专家，结果为：

    ``[[0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 6, 7], [6, 7, 0, 1]]``

    其中每行前两个值是主专家，后两个值是已在初始加载阶段就可用的冗余副本。
    """
    assert num_logical_experts % num_ranks == 0
    num_experts_per_rank = num_logical_experts // num_ranks
    assert 0 <= num_redundant_experts_per_rank <= num_logical_experts - num_experts_per_rank

    local_expert_ids_by_rank = []
    for rank in range(num_ranks):
        first_expert_id = rank * num_experts_per_rank
        local_expert_ids = list(range(first_expert_id, first_expert_id + num_experts_per_rank))
        first_redundant_expert_id = ((rank + 1) * num_experts_per_rank) % num_logical_experts
        local_expert_ids.extend(
            (first_redundant_expert_id + offset) % num_logical_experts
            for offset in range(num_redundant_experts_per_rank)
        )
        local_expert_ids_by_rank.append(local_expert_ids)

    return local_expert_ids_by_rank


def build_logical_to_physical_map(
    rank_to_logic_expert_ids: list[list[int]],
    num_logical_experts: int,
    current_rank: int,
) -> list[list[int]]:
    """使用普通 CPU list 构建单层 logical 到 physical expert 的路由表。

    ``rank_to_logic_expert_ids`` 的 shape 为
    ``[num_ranks, num_physical_experts_per_rank]``，每行包含该 rank 的全部
    主专家和冗余专家。

    返回值的 shape 为 ``[num_logical_experts, 2 + routing_slots]``。每一行
    对应一个 logical expert：第 0 项是有效副本数，第 1 项
    标记 ``current_rank`` 是否持有本地副本，第 2 项起是 physical expert ID。
    如果本 rank 持有副本，该副本固定放在第一个路由槽；有效副本之后未使用
    的固定宽度 padding 槽位填充为 ``-1``。

    本函数只负责 CPU 元数据计算。调用方需要设备 Tensor 时，应在函数外
    显式执行 ``torch.tensor(...)``。
    """
    # 阶段 1：校验输入布局，并根据每个 rank 的物理槽位数计算冗余容量。
    num_ranks = len(rank_to_logic_expert_ids)
    assert num_ranks > 0
    assert num_logical_experts % num_ranks == 0
    num_physical_experts_per_rank = len(rank_to_logic_expert_ids[0])
    assert all(len(rank_expert_ids) == num_physical_experts_per_rank for rank_expert_ids in rank_to_logic_expert_ids)
    num_primary_experts_per_rank = num_logical_experts // num_ranks
    num_redundant_experts_per_rank = num_physical_experts_per_rank - num_primary_experts_per_rank
    assert num_redundant_experts_per_rank >= 0
    # 阶段 2：计算固定路由槽宽度。最坏情况下，所有 rank 的全部冗余槽都
    # 指向同一个 logical expert；再加上该 expert 固有的一个主副本，就是
    # 任意 logical expert 可能拥有的最大物理副本数。
    num_routing_slots = 1 + num_ranks * num_redundant_experts_per_rank
    assert 0 <= current_rank < num_ranks

    # 阶段 3：把“物理槽 -> logical expert”的完整布局反转为
    # “logical expert -> 全部物理槽”，得到每个专家的候选副本列表。
    physical_ids_by_logical_expert = _collect_physical_ids_by_logical_expert(
        rank_to_logic_expert_ids,
        num_logical_experts,
    )

    # 阶段 4：对每个候选列表做稳定排序。本 rank 的 physical ID 排在前面，
    # 因而后续只需查看第一个候选，就能判断和选择本地副本。
    _sort_physical_ids_by_locality(
        physical_ids_by_logical_expert,
        current_rank,
        num_physical_experts_per_rank,
    )
    local_physical_id_start = current_rank * num_physical_experts_per_rank
    local_physical_id_end = local_physical_id_start + num_physical_experts_per_rank

    # 阶段 5：逐个 logical expert 打包固定宽度的路由行。实际副本不足固定
    # 宽度时，剩余槽位使用 -1 padding；kernel 只会索引有效副本范围。
    logical_to_physical_map = []
    for physical_expert_ids in physical_ids_by_logical_expert:
        has_local_replica = local_physical_id_start <= physical_expert_ids[0] < local_physical_id_end
        logical_to_physical_map.append(
            _build_routing_row(
                physical_expert_ids=physical_expert_ids,
                has_local_replica=has_local_replica,
                num_routing_slots=num_routing_slots,
            )
        )
    return logical_to_physical_map


def _collect_physical_ids_by_logical_expert(
    rank_to_logic_expert_ids: list[list[int]],
    num_logical_experts: int,
) -> list[list[int]]:
    """将完整物理布局反转为每个 logical expert 对应的物理槽位。

    例如输入 ``[[0, 1, 1], [2, 3, 0]]``，先按 rank 顺序拼成
    ``[0, 1, 1, 2, 3, 0]``。该列表的下标就是 physical expert ID，值就是
    logical expert ID，因此最终返回 ``[[0, 5], [1, 2], [3], [4]]``。
    """
    logical_expert_ids_by_physical_id = [
        logical_expert_id for rank_expert_ids in rank_to_logic_expert_ids for logical_expert_id in rank_expert_ids
    ]
    physical_ids_by_logical_expert = [[] for _ in range(num_logical_experts)]
    for physical_expert_id, logical_expert_id in enumerate(logical_expert_ids_by_physical_id):
        assert 0 <= logical_expert_id < num_logical_experts
        physical_ids_by_logical_expert[logical_expert_id].append(physical_expert_id)

    return physical_ids_by_logical_expert


def _sort_physical_ids_by_locality(
    physical_ids_by_logical_expert: list[list[int]],
    current_rank: int,
    num_physical_experts_per_rank: int,
) -> None:
    """按照 physical ID 是否属于当前 rank，对每个副本列表稳定排序。

    本地 physical ID 的排序键为 0，其他 physical ID 的排序键为 1。因此
    当前 rank 持有的副本会移动到列表前面，同时本地副本之间、远端副本
    之间的原始顺序保持不变。当前 rank 没有副本的列表顺序不会发生变化。
    """
    local_physical_id_start = current_rank * num_physical_experts_per_rank
    local_physical_id_end = local_physical_id_start + num_physical_experts_per_rank

    for physical_expert_ids in physical_ids_by_logical_expert:
        # list.sort 是稳定排序：排序键相同时，physical ID 的原始顺序不变。
        physical_expert_ids.sort(
            key=lambda physical_expert_id: (
                0 if local_physical_id_start <= physical_expert_id < local_physical_id_end else 1
            )
        )


def _build_routing_row(
    physical_expert_ids: list[int],
    has_local_replica: bool,
    num_routing_slots: int,
) -> list[int]:
    """将一个 logical expert 的候选 physical IDs 打包为固定宽度路由行。

    ``physical_expert_ids`` 已由调用方完成本地优先的稳定排序，所以本函数
    不再依赖 ``current_rank``。列表长度就是该 logical expert 的有效物理
    副本数，无需额外传入容易失配的副本数量。
    """
    # 阶段 1：候选列表包含一个主副本及全部冗余副本，其长度就是有效副本数。
    num_valid_replicas = len(physical_expert_ids)
    assert 0 < num_valid_replicas <= num_routing_slots

    # 阶段 2：有效槽位直接保存稳定排序后的候选；固定宽度中未使用的尾部
    # 槽位统一填充 -1。kernel 的副本索引严格小于 num_valid_replicas，
    # 因而不会读取 padding。
    num_padding_slots = num_routing_slots - num_valid_replicas
    routing_slots = physical_expert_ids + [-1] * num_padding_slots

    # 阶段 3：第 0 列保存 kernel 参与 hash 的有效副本数；第 1 列标记是否
    # 存在本地副本；后续列保存按本地优先顺序排列的 physical IDs 和 -1 padding。
    return [num_valid_replicas, int(has_local_replica), *routing_slots]


def build_logical_to_physical_maps_for_layers(
    rank_to_logic_expert_ids_by_layer: list[list[list[int]]],
    num_logical_experts: int,
    current_rank: int,
) -> list[list[list[int]]]:
    """逐层构建 CPU list 路由表；设备 Tensor 由调用方在边界处创建。

    输入 shape 为 ``[num_layers, num_ranks, num_physical_experts_per_rank]``，
    输出 shape 为 ``[num_layers, num_logical_experts, 2 + routing_slots]``。
    """
    return [
        build_logical_to_physical_map(
            rank_to_logic_expert_ids,
            num_logical_experts,
            current_rank=current_rank,
        )
        for rank_to_logic_expert_ids in rank_to_logic_expert_ids_by_layer
    ]


def select_improving_placements(
    expert_load: torch.Tensor,
    current_placement: torch.Tensor,
    candidate_placement: torch.Tensor,
    *,
    rebalance_gain_threshold: float,
    expert_alignment: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float | int], torch.Tensor, torch.Tensor]:
    """Select better layers and return current/final rank loads without re-estimation."""
    if not 0.0 <= rebalance_gain_threshold <= 1.0:
        raise ValueError("rebalance_gain_threshold must be between 0.0 and 1.0")
    assert current_placement.shape == candidate_placement.shape
    current_rank_load = _estimate_rank_load(expert_load, current_placement, expert_alignment)
    candidate_rank_load = _estimate_rank_load(expert_load, candidate_placement, expert_alignment)
    current_critical = current_rank_load.max(dim=-1).values.sum(dim=0)
    candidate_critical = candidate_rank_load.max(dim=-1).values.sum(dim=0)
    # Each changed layer must reduce its own critical load. All selected
    # changes must then collectively meet the configured model-level
    # critical-load reduction threshold, avoiding low-gain migrations.
    improved = candidate_critical < current_critical
    selected = current_placement.clone()
    selected[improved] = candidate_placement[improved]
    selected_rank_load = torch.where(improved[:, None], candidate_rank_load, current_rank_load)
    model_current_critical = current_critical.sum()
    model_current_mean = current_rank_load.mean(dim=-1).sum()
    model_selected_critical = selected_rank_load.max(dim=-1).values.sum()
    model_selected_mean = selected_rank_load.mean(dim=-1).sum()
    model_ratio = model_current_critical / model_current_mean.clamp_min(1.0)
    candidate_model_ratio = model_selected_critical / model_selected_mean.clamp_min(1.0)
    candidate_rebalance_gain = (model_current_critical - model_selected_critical) / model_current_critical.clamp_min(
        1.0
    )
    metrics = {
        "model_imbalance_ratio": float(model_ratio.item()),
        "candidate_model_imbalance_ratio": float(candidate_model_ratio.item()),
        "candidate_rebalance_gain": float(candidate_rebalance_gain.item()),
        "candidate_changed_layer_count": int(improved.sum().item()),
    }
    if candidate_rebalance_gain >= rebalance_gain_threshold:
        return selected, improved, metrics, current_rank_load, selected_rank_load
    return (
        current_placement.clone(),
        torch.zeros_like(improved),
        metrics,
        current_rank_load,
        current_rank_load,
    )


def plan_redundant_experts(
    expert_load: torch.Tensor,
    num_ranks: int,
    num_redundant_experts_per_rank: int,
    expert_alignment: int | None = None,
    current_placement: torch.Tensor | None = None,
    stickiness: float = 0.0,
) -> torch.Tensor:
    """Plan replicas from [samples, layers, current_ranks, experts] loads.

    With ``current_placement`` and positive ``stickiness``, a candidate that
    keeps an expert on its current rank receives a bonus of
    ``stickiness * mean per-layer expert load``. This preserves rank
    membership, not a particular redundant physical slot; target slots are
    canonicalized against the current live rows before transfer and metadata
    publication. A rank membership only changes when the move improves the
    critical-load objective by more than that margin.
    With zero stickiness, placement is determined solely by the load objective.
    """
    if expert_alignment is not None:
        assert expert_alignment > 0
    assert expert_load.ndim == 4
    _, num_layers, num_load_ranks, num_logical_experts = expert_load.shape
    assert num_load_ranks in (1, num_ranks)
    assert num_logical_experts % num_ranks == 0
    assert num_redundant_experts_per_rank > 0
    num_experts_per_rank = num_logical_experts // num_ranks
    num_redundant = num_ranks * num_redundant_experts_per_rank
    assert num_redundant <= num_logical_experts * (num_ranks - 1)

    load = expert_load.to(dtype=torch.float64, device="cpu")
    placement = torch.full((num_layers, num_ranks, num_redundant_experts_per_rank), -1, dtype=torch.int64)
    owner_rank = torch.arange(num_logical_experts, dtype=torch.int64) // num_experts_per_rank
    if current_placement is not None:
        assert tuple(current_placement.shape) == (
            num_layers,
            num_ranks,
            num_redundant_experts_per_rank,
        )
        current_locations = _expert_locations(current_placement, num_logical_experts)
        stickiness_scale = load.sum(dim=(0, 2, 3)) / num_logical_experts
    else:
        current_locations = None
        stickiness_scale = None

    locations = _expert_locations(placement, num_logical_experts)
    expert_rank = _expert_rank_load_all(load, locations, expert_alignment)
    rank_load = expert_rank.sum(dim=2)
    remaining_slots = torch.full((num_layers, num_ranks), num_redundant_experts_per_rank, dtype=torch.int64)
    layer_indices = torch.arange(num_layers, dtype=torch.int64)
    expert_ids = torch.arange(num_logical_experts, dtype=torch.int64)
    # Every iteration fills one slot per layer.  Candidate expert evaluation
    # is vectorized across all layers and logical experts, which keeps large
    # GLM/Qwen planning comfortably on the CPU fast path.
    for _ in range(num_redundant):
        rank_order = torch.argsort(rank_load.sum(dim=0), dim=1, stable=True)
        target_ranks = torch.full((num_layers,), -1, dtype=torch.int64)
        legal = torch.zeros((num_layers, num_logical_experts), dtype=torch.bool)
        for layer in range(num_layers):
            for target_rank in rank_order[layer].tolist():
                if remaining_slots[layer, target_rank] == 0:
                    continue
                candidate_legal = (owner_rank != target_rank) & ~locations[layer, :, target_rank]
                if torch.any(candidate_legal):
                    target_ranks[layer] = target_rank
                    legal[layer] = candidate_legal
                    break
        if torch.any(target_ranks < 0):
            raise RuntimeError("EPLB planner found no valid redundant expert placement")

        candidate_locations = locations.clone()
        candidate_locations[layer_indices[:, None], expert_ids[None, :], target_ranks[:, None]] = True
        candidate_expert_rank = _expert_rank_load_all(load, candidate_locations, expert_alignment)
        candidate_rank_load = rank_load[:, :, None, :] - expert_rank + candidate_expert_rank
        critical = candidate_rank_load.max(dim=3).values.sum(dim=0)
        critical.masked_fill_(~legal, torch.inf)
        if current_locations is not None:
            # An expert already held by the target rank is retained unless
            # another candidate beats it by more than the stickiness margin.
            # This is rank membership, not physical-slot stickiness. Masked
            # (inf) candidates stay masked: inf - x == inf.
            keep = current_locations[layer_indices[:, None], expert_ids[None, :], target_ranks[:, None]]
            critical = critical - stickiness * stickiness_scale[:, None] * keep
        selected_experts = critical.argmin(dim=1)
        if torch.isinf(critical[layer_indices, selected_experts]).any():
            raise RuntimeError("EPLB planner found no valid redundant expert placement")

        slots = num_redundant_experts_per_rank - remaining_slots[layer_indices, target_ranks]
        placement[layer_indices, target_ranks, slots] = selected_experts
        selected_next = candidate_expert_rank[:, layer_indices, selected_experts]
        selected_old = expert_rank[:, layer_indices, selected_experts]
        rank_load += selected_next - selected_old
        expert_rank[:, layer_indices, selected_experts] = selected_next
        locations[layer_indices, selected_experts, target_ranks] = True
        remaining_slots[layer_indices, target_ranks] -= 1

    assert torch.all(placement >= 0)
    return placement


def _estimate_rank_load(
    expert_load: torch.Tensor,
    rank_to_logic_expert_ids: torch.Tensor,
    expert_alignment: int | None = None,
) -> torch.Tensor:
    """Estimate [samples, layers, ranks] load from current-rank-local routing.

    Per-rank loads remain separate until assigned to physical replicas, then
    combine before the per-expert alignment used by DeepEP.
    """
    assert expert_load.ndim == 4
    _, num_layers, num_load_ranks, num_logical_experts = expert_load.shape
    assert rank_to_logic_expert_ids.ndim == 3 and rank_to_logic_expert_ids.shape[0] == num_layers
    num_ranks = rank_to_logic_expert_ids.shape[1]
    assert num_load_ranks in (1, num_ranks)
    assert num_logical_experts % num_ranks == 0
    if expert_alignment is not None:
        assert expert_alignment > 0

    rank_load = _expert_rank_load_all(
        expert_load,
        _expert_locations(rank_to_logic_expert_ids, num_logical_experts),
        expert_alignment,
    ).sum(dim=2)
    return rank_load


def _expert_locations(rank_to_logic_expert_ids: torch.Tensor, num_logical_experts: int) -> torch.Tensor:
    """Return ``[layer, logical expert, rank]`` physical-copy occupancy."""
    num_layers, num_ranks, num_redundant_experts_per_rank = rank_to_logic_expert_ids.shape
    assert num_logical_experts % num_ranks == 0
    num_experts_per_rank = num_logical_experts // num_ranks
    locations = torch.zeros(
        (num_layers, num_logical_experts, num_ranks),
        dtype=torch.bool,
        device=rank_to_logic_expert_ids.device,
    )
    expert_ids = torch.arange(num_logical_experts, device=locations.device)
    owners = expert_ids // num_experts_per_rank
    locations[:, expert_ids, owners] = True
    layers = torch.arange(num_layers, device=locations.device)[:, None]
    ranks = torch.arange(num_ranks, device=locations.device).repeat_interleave(num_redundant_experts_per_rank)[None, :]
    flat_logic_expert_ids = rank_to_logic_expert_ids.reshape(num_layers, -1)
    valid = flat_logic_expert_ids >= 0
    if torch.any(valid):
        expanded_layers = layers.expand_as(flat_logic_expert_ids)
        expanded_ranks = ranks.expand_as(flat_logic_expert_ids)
        locations[
            expanded_layers[valid],
            flat_logic_expert_ids[valid],
            expanded_ranks[valid],
        ] = True
    return locations


def _current_rank_route(slots: torch.Tensor, num_load_ranks: int) -> torch.Tensor:
    """当前 rank 有本地副本时只选本地，否则在所有副本之间均分。

    ``num_load_ranks == 1`` 表示离线调用方只提供了聚合负载，此时无法判断
    当前 rank，直接在全部副本之间均分。线上采样始终传入逐 rank 负载。
    """
    num_ranks = slots.shape[-1]
    copies = slots.unsqueeze(-3).expand(*slots.shape[:-2], num_load_ranks, *slots.shape[-2:])
    if num_load_ranks == 1:
        return copies.to(torch.float64) / copies.sum(dim=-1, keepdim=True)

    assert num_load_ranks == num_ranks
    ranks = torch.arange(num_ranks, device=slots.device)
    destination_rank_shape = (1,) * slots.ndim + (num_ranks,)
    current_rank_shape = (1,) * (slots.ndim - 2) + (num_ranks, 1, 1)
    local = copies & (ranks.reshape(destination_rank_shape) == ranks.reshape(current_rank_shape))
    selected = torch.where(local.any(dim=-1, keepdim=True), local, copies)
    return selected.to(torch.float64) / selected.sum(dim=-1, keepdim=True)


def _expert_rank_load_all(
    expert_load: torch.Tensor,
    locations: torch.Tensor,
    expert_alignment: int | None,
) -> torch.Tensor:
    """Return aligned ``[samples, layers, expert, rank]`` contributions."""
    route = _current_rank_route(locations, expert_load.shape[2])
    physical_load = torch.einsum("slqe,lqer->sler", expert_load.to(torch.float64), route)
    if expert_alignment is not None:
        physical_load = torch.ceil(physical_load / expert_alignment) * expert_alignment
    return physical_load
