from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Tuple
import torch


def build_initial_redundant_expert_ids(
    num_logical_experts: int,
    num_ranks: int,
    num_redundant_experts_per_rank: int,
) -> torch.Tensor:
    """Build a deterministic initial placement without local duplicates."""
    assert num_logical_experts % num_ranks == 0
    num_experts_per_rank = num_logical_experts // num_ranks
    assert 0 < num_redundant_experts_per_rank <= num_logical_experts - num_experts_per_rank

    # 初始化结果确定，不依赖随机数。
    # 每个 rank 不会复制自己原本拥有的 expert。
    # 同一个 rank 的冗余槽位不会重复。
    # 最后一个 rank 通过取模自然回绕。
    rank_offsets = torch.arange(1, num_ranks + 1, dtype=torch.int64)[:, None] * num_experts_per_rank
    expert_offsets = torch.arange(num_redundant_experts_per_rank, dtype=torch.int64)
    return (rank_offsets + expert_offsets) % num_logical_experts


def build_logical_to_physical_map(
    redundant_expert_ids: torch.Tensor,  # 冗余布局，shape 为 [num_ranks, num_redundant_experts_per_rank]。
    num_logical_experts: int,  # 逻辑 expert 的总数。
    source_rank: int | None = None,  # 可选的全局源 rank；传入时优先选择同节点副本。
    node_world_size: int | None = None,  # 单个节点包含的 rank 数；source_rank 非空时必填。
) -> Tuple[
    torch.Tensor, torch.Tensor
]:  # logical_to_physical [num_logical_experts, num_ranks], replica_counts [num_logical_experts]
    """构建单层逻辑 expert 到物理副本的映射"""

    logical_to_physical, replica_counts = _build_layer_maps(
        redundant_expert_ids.unsqueeze(0),
        num_logical_experts,
        source_rank=source_rank,
        node_world_size=node_world_size,
    )
    return logical_to_physical.squeeze(0), replica_counts.squeeze(0)


def build_logical_to_physical_maps_for_layers(
    redundant_expert_ids_by_layer: torch.Tensor,  # [num_layers, num_ranks, num_redundant_experts_per_rank]
    num_logical_experts: int,  # 逻辑 expert 的总数。
    source_rank: int | None = None,  # 可选的全局源 rank；传入时优先选择同节点副本。
    node_world_size: int | None = None,  # 单个节点包含的 rank 数；source_rank 非空时必填。
) -> Tuple[
    torch.Tensor,  # logical_to_physical, shape [num_layers, num_logical_experts, num_ranks]
    torch.Tensor,  # replica_counts, shape [num_layers, num_logical_experts]
]:
    """为调用方传入的多个指定层构建逻辑 expert 到物理副本的 CPU int32 映射。

    第一维是层，不是请求或 token 的 batch，也不会自动处理模型中的其他层。
    全局构建阶段第 0 列固定为主副本，其余列按 rank-major、slot-major 的稳定顺序
    写入冗余副本。指定 source_rank 后，先筛选源节点内副本，再按 source_rank
    轮转候选前缀；最终返回映射的第 0 列只是第一个候选，不保证仍是主副本。
    """
    return _build_layer_maps(
        redundant_expert_ids_by_layer,
        num_logical_experts,
        source_rank=source_rank,
        node_world_size=node_world_size,
    )


def select_improving_placements(
    expert_load: torch.Tensor,
    current_placement: torch.Tensor,
    candidate_placement: torch.Tensor,
    *,
    rebalance_gain_threshold: float,
    expert_alignment: int | None = None,
    node_world_size: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float | int], torch.Tensor, torch.Tensor]:
    """Select better layers and return current/final rank loads without re-estimation."""
    if not 0.0 <= rebalance_gain_threshold <= 1.0:
        raise ValueError("rebalance_gain_threshold must be between 0.0 and 1.0")
    assert current_placement.shape == candidate_placement.shape
    current_rank_load = _estimate_rank_load(expert_load, current_placement, expert_alignment, node_world_size)
    candidate_rank_load = _estimate_rank_load(expert_load, candidate_placement, expert_alignment, node_world_size)
    if expert_load.ndim == 2:
        current_critical = current_rank_load.max(dim=1).values
        candidate_critical = candidate_rank_load.max(dim=1).values
    else:
        current_critical = current_rank_load.max(dim=2).values.sum(dim=0)
        candidate_critical = candidate_rank_load.max(dim=2).values.sum(dim=0)
    # Each changed layer must reduce its own critical load. All selected
    # changes must then collectively meet the configured model-level
    # critical-load reduction threshold, avoiding low-gain migrations.
    improved = candidate_critical < current_critical
    selected = current_placement.clone()
    selected[improved] = candidate_placement[improved]
    if current_rank_load.ndim == 2:
        selected_rank_load = torch.where(improved[:, None], candidate_rank_load, current_rank_load)
    else:
        selected_rank_load = torch.where(improved[None, :, None], candidate_rank_load, current_rank_load)
    model_current_critical = current_critical.sum()
    if expert_load.ndim == 2:
        model_current_mean = current_rank_load.mean(dim=1).sum()
        model_selected_critical = selected_rank_load.max(dim=1).values.sum()
        model_selected_mean = selected_rank_load.mean(dim=1).sum()
    else:
        model_current_mean = current_rank_load.mean(dim=2).sum()
        model_selected_critical = selected_rank_load.max(dim=2).values.sum()
        model_selected_mean = selected_rank_load.mean(dim=2).sum()
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
    node_world_size: int | None = None,
    current_placement: torch.Tensor | None = None,
    stickiness: float = 0.0,
) -> torch.Tensor:
    """Plan replicas using source-node-local copies, with global fallback.

    With ``current_placement`` and a positive ``stickiness``, a candidate that
    keeps an expert on its current rank receives a bonus of
    ``stickiness * mean per-layer expert load``. This preserves rank
    membership, not a particular redundant physical slot; target slots are
    canonicalized against the current live rows before transfer and metadata
    publication. A rank membership only changes when the move improves the
    critical-load objective by more than that margin.
    Without them the planning is bit-identical to the legacy behavior.
    """
    assert expert_load.ndim in (2, 3, 4)
    if expert_alignment is not None:
        assert expert_alignment > 0
    use_legacy_topology_preference = expert_load.ndim < 4
    legacy_node_world_size = node_world_size if use_legacy_topology_preference else None
    source_load, _squeeze_sample, node_world_size = _as_source_node_load(expert_load, num_ranks, node_world_size)
    num_samples, num_layers, num_nodes, num_logical_experts = source_load.shape
    assert num_logical_experts % num_ranks == 0
    assert num_redundant_experts_per_rank > 0
    num_experts_per_rank = num_logical_experts // num_ranks
    num_redundant = num_ranks * num_redundant_experts_per_rank
    assert num_redundant <= num_logical_experts * (num_ranks - 1)

    load = source_load.to(dtype=torch.float64, device="cpu")
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
    expert_rank = _expert_rank_load_all(load, locations, num_nodes, node_world_size, expert_alignment)
    rank_load = expert_rank.sum(dim=2)
    remaining_slots = torch.full((num_layers, num_ranks), num_redundant_experts_per_rank, dtype=torch.int64)
    layer_indices = torch.arange(num_layers, dtype=torch.int64)
    expert_ids = torch.arange(num_logical_experts, dtype=torch.int64)
    rank_nodes = (
        torch.arange(num_ranks, dtype=torch.int64) // legacy_node_world_size
        if legacy_node_world_size is not None and legacy_node_world_size < num_ranks
        else None
    )

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
                # Legacy 2D/3D callers have no source-node axis.  Retain the
                # previous topology preference for that compatibility path;
                # node-aware [S,L,N,E] planning uses only the exact load
                # objective below.
                if rank_nodes is not None:
                    existing_on_target_node = locations[layer, :, rank_nodes == rank_nodes[target_rank]].any(dim=1)
                    new_node_legal = candidate_legal & ~existing_on_target_node
                    if torch.any(new_node_legal):
                        candidate_legal = new_node_legal
                if torch.any(candidate_legal):
                    target_ranks[layer] = target_rank
                    legal[layer] = candidate_legal
                    break
        if torch.any(target_ranks < 0):
            raise RuntimeError("EPLB planner found no valid redundant expert placement")

        candidate_locations = locations.clone()
        candidate_locations[layer_indices[:, None], expert_ids[None, :], target_ranks[:, None]] = True
        candidate_expert_rank = _expert_rank_load_all(
            load, candidate_locations, num_nodes, node_world_size, expert_alignment
        )
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


@dataclass(frozen=True, eq=False)
class _PhysicalExpertLayout:
    """进程内按拓扑复用的只读物理 expert 布局；其中 Tensor 不得原地修改。"""

    num_logical_experts: int
    num_ranks: int
    num_physical_experts_per_rank: int
    primary_physical_ids: torch.Tensor
    redundant_physical_ids: torch.Tensor


@lru_cache(maxsize=8)
def _get_physical_expert_layout(
    num_logical_experts: int,
    num_ranks: int,
    num_redundant_experts_per_rank: int,
) -> _PhysicalExpertLayout:
    """返回按静态拓扑缓存的只读 CPU 物理 expert ID。"""
    num_experts_per_rank = num_logical_experts // num_ranks
    num_physical_experts_per_rank = num_experts_per_rank + num_redundant_experts_per_rank
    expert_ids = torch.arange(num_logical_experts, dtype=torch.int64)
    primary_physical_ids = (
        (expert_ids // num_experts_per_rank) * num_physical_experts_per_rank + expert_ids % num_experts_per_rank
    ).to(torch.int32)
    ranks = torch.arange(num_ranks, dtype=torch.int64).repeat_interleave(num_redundant_experts_per_rank)
    slots = torch.arange(num_redundant_experts_per_rank, dtype=torch.int64).repeat(num_ranks)
    redundant_physical_ids = (ranks * num_physical_experts_per_rank + num_experts_per_rank + slots).to(torch.int32)
    return _PhysicalExpertLayout(
        num_logical_experts=num_logical_experts,
        num_ranks=num_ranks,
        num_physical_experts_per_rank=num_physical_experts_per_rank,
        primary_physical_ids=primary_physical_ids,
        redundant_physical_ids=redundant_physical_ids,
    )


def _build_global_replica_maps_for_layers(
    redundant_expert_ids_by_layer: torch.Tensor,  # [num_layers, num_ranks, num_redundant_experts_per_rank]
    layout: _PhysicalExpertLayout,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """为显式传入的多层冗余布局构建全局逻辑 expert id 到物理位置序列[rank, local_slot]的映射。

    输出：
        logical_to_physical：CPU ``int32`` Tensor，形状为
            ``[num_layers, num_logical_experts, num_ranks]``。第 0 列固定为主
            副本，后续列依次存放冗余副本，未使用位置为 ``-1``。
        replica_counts：CPU ``int32`` Tensor，形状为
            ``[num_layers, num_logical_experts]``。每个值包含主副本，并表示
            ``logical_to_physical`` 对应行中有效副本连续前缀的长度。

    “全局映射”记录每层每个逻辑 expert 在所有 rank 上的主副本和冗余副本所对应的
    物理 expert ID。第 0 列固定为主副本；冗余副本按所在 rank 从小到大、同一 rank
    内按槽位从小到大的顺序写入后续列。输入参数
    ``redundant_expert_ids_by_layer`` 已经指定每个 rank 的每个冗余槽位存放哪个逻辑
    expert，本函数只将该输入转换为顺序确定的映射，满足主副本优先、冗余槽位对应正确且有效副本连续排列的确定性结果。

    """

    num_layers = redundant_expert_ids_by_layer.shape[0]
    num_logical_experts = layout.num_logical_experts
    max_replicas = layout.num_ranks
    redundant_ids = redundant_expert_ids_by_layer.to(dtype=torch.int64, device="cpu")
    logical_to_physical = torch.full((num_layers, num_logical_experts, max_replicas), -1, dtype=torch.int32)
    logical_to_physical[:, :, 0] = layout.primary_physical_ids
    replica_counts = torch.ones((num_layers, num_logical_experts), dtype=torch.int32)

    flat_redundant_ids = redundant_ids.reshape(num_layers, -1)
    if not flat_redundant_ids.numel():
        return logical_to_physical, replica_counts

    # 稳定排序保留 rank-major、slot-major 的历史顺序；第 0 列固定为主副本。
    sort_order = torch.argsort(flat_redundant_ids, dim=1, stable=True)
    sorted_redundant_ids = flat_redundant_ids.gather(1, sort_order)
    flat_positions = torch.arange(flat_redundant_ids.shape[1], dtype=torch.int64).unsqueeze(0)
    group_starts = torch.where(
        torch.cat(
            (
                torch.ones((num_layers, 1), dtype=torch.bool),
                sorted_redundant_ids[:, 1:] != sorted_redundant_ids[:, :-1],
            ),
            dim=1,
        ),
        flat_positions,
        0,
    )
    replica_indices = flat_positions - torch.cummax(group_starts, dim=1).values + 1
    redundant_counts = torch.zeros((num_layers, num_logical_experts), dtype=torch.int32)
    redundant_counts.scatter_add_(
        1,
        flat_redundant_ids,
        torch.ones_like(flat_redundant_ids, dtype=torch.int32),
    )
    assert int(redundant_counts.max().item()) < max_replicas, "an expert can have at most one replica per rank"
    replica_counts += redundant_counts

    layer_indices = torch.arange(num_layers, dtype=torch.int64).view(-1, 1).expand_as(sort_order)
    redundant_physical_ids = layout.redundant_physical_ids.unsqueeze(0).expand_as(sort_order).gather(1, sort_order)
    logical_to_physical[layer_indices, sorted_redundant_ids, replica_indices] = redundant_physical_ids
    return logical_to_physical, replica_counts


def _select_source_node_replicas(
    logical_to_physical: torch.Tensor,
    replica_counts: torch.Tensor,
    *,
    source_rank: int,
    node_world_size: int,
    num_physical_experts_per_rank: int,
    replica_positions: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """按来源节点筛选全局候选副本，并将结果压缩为连续前缀。

    ``logical_to_physical[layer, logical_expert]`` 的前 ``replica_counts`` 个位置
    是有效副本，尾部 ``-1`` 表示没有副本；正常输入和输出都不会在有效前缀中
    出现 ``-1``。连续的 ``node_world_size`` 个 rank 构成一个节点，来源节点为
    ``source_rank // node_world_size``。对每个逻辑 expert，若来源节点有副本，
    则只保留该节点的全部副本，远程副本（包括远程主副本）全部排除；否则保留
    所有全局有效副本作为回退，避免候选为空。筛选可能选中原有效前缀中不连续的
    位置，因此按原相对顺序将选中副本复制到新映射的连续前缀，其余位置填 ``-1``，
    并返回新的有效数量；不会原地修改输入。此函数只筛选和压缩候选集合，不做
    负载优化、最终副本选择或按 source rank 轮转，轮转由后续函数完成。

    输出：
      - ``compact_maps_by_layer``：与 ``logical_to_physical`` 同 shape
        ``[num_layers, num_logical_experts, num_ranks]``，dtype/device 相同；每行
        为筛选后副本的连续有效前缀，尾部为 ``-1``。
      - ``selected_counts_by_layer``：输入同 device 的 ``int32`` Tensor，shape 为
        ``[num_layers, num_logical_experts]``；每个值是对应输出行的有效前缀长度。
    """
    num_layers, num_logical_experts, _max_replicas = logical_to_physical.shape
    source_node = source_rank // node_world_size
    output_positions = replica_positions.view(1, 1, -1)
    valid = output_positions < replica_counts.unsqueeze(-1)
    local = valid & (
        torch.div(
            logical_to_physical,
            num_physical_experts_per_rank * node_world_size,
            rounding_mode="floor",
        )
        == source_node
    )
    selected = torch.where(local.any(dim=2, keepdim=True), local, valid)
    selected_counts_by_layer = selected.sum(dim=2, dtype=torch.int32)

    compact_maps_by_layer = torch.full_like(logical_to_physical, -1)
    selected_positions = selected.cumsum(dim=2) - 1
    layers = torch.arange(num_layers, dtype=torch.int64).view(-1, 1, 1).expand_as(selected)
    experts = torch.arange(num_logical_experts, dtype=torch.int64).view(1, -1, 1).expand_as(selected)
    compact_maps_by_layer[layers[selected], experts[selected], selected_positions[selected]] = logical_to_physical[
        selected
    ]
    return compact_maps_by_layer, selected_counts_by_layer


def _rotate_selected_replicas(
    compact_maps_by_layer: torch.Tensor,
    selected_count_by_layer: torch.Tensor,
    *,
    source_rank: int,
    replica_positions: torch.Tensor,
) -> torch.Tensor:
    """根据 source_rank 循环调整每个逻辑 expert 的候选副本顺序，让不同源 rank 优先使用不同副本，同时保持候选副本集合和副本数量不变"""
    output_positions = replica_positions.view(1, 1, -1)
    selected_count64_by_layer = selected_count_by_layer.to(torch.int64).unsqueeze(-1)
    rotation_by_layer = source_rank % selected_count64_by_layer
    source_positions_by_layer = (output_positions + rotation_by_layer) % selected_count64_by_layer
    maps_by_layer = compact_maps_by_layer.gather(2, source_positions_by_layer)
    maps_by_layer.masked_fill_(output_positions >= selected_count64_by_layer, -1)
    return maps_by_layer


def _build_layer_maps(
    redundant_expert_ids_by_layer: torch.Tensor,
    num_logical_experts: int,  # 逻辑 expert 的总数。
    source_rank: int | None = None,  # 可选的全局源 rank；传入时优先选择同节点副本。
    node_world_size: int | None = None,  # 单个节点包含的 rank 数；source_rank 非空时必填。
) -> Tuple[torch.Tensor, torch.Tensor]:
    """构建调用方指定层的映射；全局构建、节点筛选和轮转均在此完成。"""
    if redundant_expert_ids_by_layer.ndim != 3:
        raise ValueError("redundant_expert_ids_by_layer must be [layers, ranks, num_redundant_experts_per_rank]")
    num_ranks, num_redundant_experts_per_rank = redundant_expert_ids_by_layer.shape[1:]
    assert num_logical_experts % num_ranks == 0
    layout = _get_physical_expert_layout(num_logical_experts, num_ranks, num_redundant_experts_per_rank)
    logical_to_physical, replica_counts = _build_global_replica_maps_for_layers(redundant_expert_ids_by_layer, layout)
    if source_rank is None:
        return logical_to_physical, replica_counts

    assert node_world_size is not None
    replica_positions = torch.arange(num_ranks, dtype=torch.int64)
    compact_maps_by_layer, selected_counts_by_layer = _select_source_node_replicas(
        logical_to_physical,
        replica_counts,
        source_rank=source_rank,
        node_world_size=node_world_size,
        num_physical_experts_per_rank=layout.num_physical_experts_per_rank,
        replica_positions=replica_positions,
    )
    return (
        _rotate_selected_replicas(
            compact_maps_by_layer,
            selected_counts_by_layer,
            source_rank=source_rank,
            replica_positions=replica_positions,
        ),
        selected_counts_by_layer,
    )


def _estimate_rank_load(
    expert_load: torch.Tensor,
    redundant_expert_ids: torch.Tensor,
    expert_alignment: int | None = None,
    node_world_size: int | None = None,
) -> torch.Tensor:
    """Estimate runtime source-node-local routing load per physical expert.

    ``expert_load`` accepts the historic ``[layers, experts]`` and
    ``[samples, layers, experts]`` forms, which are both one source node, and
    the distributed ``[samples, layers, source_nodes, experts]`` form.  Source
    loads are kept separate until they are assigned to physical replicas, then
    combined before applying the per-expert alignment used by DeepEP.
    """
    source_load, squeeze_sample, node_world_size = _as_source_node_load(
        expert_load, redundant_expert_ids.shape[1], node_world_size
    )
    num_samples, num_layers, num_nodes, num_logical_experts = source_load.shape
    assert redundant_expert_ids.ndim == 3 and redundant_expert_ids.shape[0] == num_layers
    num_ranks, num_redundant_experts_per_rank = redundant_expert_ids.shape[1:]
    assert num_logical_experts % num_ranks == 0
    if expert_alignment is not None:
        assert expert_alignment > 0

    locations = _expert_locations(redundant_expert_ids, num_logical_experts)
    route = _source_route(locations, num_nodes, node_world_size)
    physical_load = torch.einsum("slne,lner->sler", source_load.to(torch.float64), route)
    if expert_alignment is not None:
        physical_load = torch.ceil(physical_load / expert_alignment) * expert_alignment
    rank_load = physical_load.sum(dim=2)
    return rank_load.squeeze(0) if squeeze_sample else rank_load


def _as_source_node_load(
    expert_load: torch.Tensor, num_ranks: int, node_world_size: int | None
) -> Tuple[torch.Tensor, bool, int]:
    """Normalize load to ``[samples, layers, source_nodes, experts]``."""
    assert expert_load.ndim in (2, 3, 4)
    squeeze_sample = expert_load.ndim == 2
    if expert_load.ndim == 2:
        source_load = expert_load.unsqueeze(0).unsqueeze(2)
    elif expert_load.ndim == 3:
        source_load = expert_load.unsqueeze(2)
    else:
        source_load = expert_load
    num_nodes = source_load.shape[2]
    # Historic 2D/3D loads represent one source node containing every rank.
    if expert_load.ndim < 4:
        return source_load, squeeze_sample, num_ranks
    if node_world_size is None:
        assert num_ranks % num_nodes == 0
        node_world_size = num_ranks // num_nodes
    assert 0 < node_world_size <= num_ranks and num_ranks % node_world_size == 0
    assert num_nodes == num_ranks // node_world_size
    return source_load, squeeze_sample, node_world_size


def _expert_locations(redundant_expert_ids: torch.Tensor, num_logical_experts: int) -> torch.Tensor:
    """Return ``[layer, logical expert, rank]`` physical-copy occupancy."""
    num_layers, num_ranks, num_redundant_experts_per_rank = redundant_expert_ids.shape
    assert num_logical_experts % num_ranks == 0
    num_experts_per_rank = num_logical_experts // num_ranks
    locations = torch.zeros(
        (num_layers, num_logical_experts, num_ranks),
        dtype=torch.bool,
        device=redundant_expert_ids.device,
    )
    expert_ids = torch.arange(num_logical_experts, device=locations.device)
    owners = expert_ids // num_experts_per_rank
    locations[:, expert_ids, owners] = True
    layers = torch.arange(num_layers, device=locations.device)[:, None]
    ranks = torch.arange(num_ranks, device=locations.device).repeat_interleave(num_redundant_experts_per_rank)[None, :]
    redundant_ids = redundant_expert_ids.reshape(num_layers, -1)
    valid = redundant_ids >= 0
    if torch.any(valid):
        expanded_layers = layers.expand_as(redundant_ids)
        expanded_ranks = ranks.expand_as(redundant_ids)
        locations[
            expanded_layers[valid],
            redundant_ids[valid],
            expanded_ranks[valid],
        ] = True
    return locations


def _source_route(slots: torch.Tensor, num_nodes: int, node_world_size: int) -> torch.Tensor:
    """Route each source node to its local copies, or all copies as fallback."""
    num_ranks = slots.shape[-1]
    assert num_ranks % node_world_size == 0 and num_nodes == num_ranks // node_world_size
    rank_nodes = torch.arange(num_ranks, device=slots.device) // node_world_size
    source_nodes = torch.arange(num_nodes, device=slots.device)
    copies = slots.unsqueeze(-3).expand(*slots.shape[:-2], num_nodes, *slots.shape[-2:])
    rank_node_shape = (1,) * slots.ndim + (num_ranks,)
    source_node_shape = (1,) * (slots.ndim - 2) + (num_nodes, 1, 1)
    local = copies & (rank_nodes.reshape(rank_node_shape) == source_nodes.reshape(source_node_shape))
    selected = torch.where(local.any(dim=-1, keepdim=True), local, copies)
    return selected.to(torch.float64) / selected.sum(dim=-1, keepdim=True)


def _expert_rank_load_all(
    source_load: torch.Tensor,
    locations: torch.Tensor,
    num_nodes: int,
    node_world_size: int,
    expert_alignment: int | None,
) -> torch.Tensor:
    """Return aligned ``[samples, layers, expert, rank]`` contributions."""
    route = _source_route(locations, num_nodes, node_world_size)
    physical_load = torch.einsum("slne,lner->sler", source_load.to(torch.float64), route)
    if expert_alignment is not None:
        physical_load = torch.ceil(physical_load / expert_alignment) * expert_alignment
    return physical_load
