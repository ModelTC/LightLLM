"""Build compact logical-to-physical routing metadata for one MoE layer.

The input layout uses ``[rank][local physical expert] -> logical expert``.
Initial placement construction lives in :mod:`.initial`; this module only
inverts an existing placement into the fixed-width rows consumed by the
EPLB routing kernel.
"""

from .types import LayerPlacement, LogicalToPhysicalMap


def build_logical_to_physical_map(
    rank_to_logic_expert_ids: LayerPlacement,
    num_logical_experts: int,
    current_rank: int,
    node_world_size: int,
) -> LogicalToPhysicalMap:
    """使用普通 CPU list 构建单层 logical 到 physical expert 的路由表。

    ``rank_to_logic_expert_ids`` 的 shape 为
    ``[num_ranks, num_physical_experts_per_rank]``，每行包含该 rank 的全部
    物理专家。

    返回值的 shape 为 ``[num_logical_experts, 3 + routing_slots]``。每一行的
    可视化结构如下：

    ``[global_count, node_count, current_gpu_count, physical_ids..., -1 padding...]``

    * ``global_count``：所有 rank 上的有效副本总数；
    * ``node_count``：当前节点上的有效副本数，包含本卡副本；
    * ``current_gpu_count``：当前 GPU 上的有效副本数；
    * ``physical_ids``：依次按本卡、本节点其他卡、其他节点排列的副本 ID。

    路由时优先使用最靠近当前 GPU 的非空候选集合：先使用本卡副本，其次使用
    本节点副本，当前节点没有副本时才使用所有 rank 的副本。有效副本之后未
    使用的固定宽度槽位填充为 ``-1``。

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
    # 阶段 2：使用整个 world 的物理槽位总数作为固定路由槽宽度。实际候选
    # 仍只写入有效副本，其余槽位统一 padding 为 -1。
    num_routing_slots = num_ranks * num_physical_experts_per_rank
    assert 0 <= current_rank < num_ranks
    assert 0 < node_world_size <= num_ranks
    assert num_ranks % node_world_size == 0

    # 阶段 3：把“物理槽 -> logical expert”的完整布局反转为
    # “logical expert -> 全部物理槽”，得到每个专家的候选副本列表。
    physical_ids_by_logical_expert = _collect_physical_ids_by_logical_expert(
        rank_to_logic_expert_ids,
        num_logical_experts,
    )

    # 阶段 4：对每个候选列表做稳定排序。当前 rank 的 physical ID 排在最前，
    # 同节点其他 rank 次之，跨节点副本最后。
    _sort_physical_ids_by_locality(
        physical_ids_by_logical_expert,
        current_rank,
        num_physical_experts_per_rank,
        node_world_size,
    )
    # 阶段 5：逐个 logical expert 打包固定宽度的路由行。实际副本不足固定
    # 宽度时，剩余槽位使用 -1 padding；kernel 只会索引有效副本范围。
    logical_to_physical_map = []
    current_node = current_rank // node_world_size
    current_node_rank_start = current_node * node_world_size
    current_node_rank_end = current_node_rank_start + node_world_size
    for physical_expert_ids in physical_ids_by_logical_expert:
        replica_ranks = [
            physical_expert_id // num_physical_experts_per_rank for physical_expert_id in physical_expert_ids
        ]
        num_node_replicas = sum(
            current_node_rank_start <= replica_rank < current_node_rank_end for replica_rank in replica_ranks
        )
        num_current_gpu_replicas = sum(replica_rank == current_rank for replica_rank in replica_ranks)
        logical_to_physical_map.append(
            _build_routing_row(
                physical_expert_ids=physical_expert_ids,
                num_node_replicas=num_node_replicas,
                num_current_gpu_replicas=num_current_gpu_replicas,
                num_routing_slots=num_routing_slots,
            )
        )
    return logical_to_physical_map


def build_logical_to_rank_physical_map(
    rank_to_logic_expert_ids: LayerPlacement, num_logical_experts: int
) -> list[list[int]]:
    """构建按 rank 查询的副本表 [E, W]，缺少副本的位置填 -1。

    输入为单层布局 [rank, local_slot]，返回普通 CPU list。
    logical_to_rank_physical_map[logical_id][rank] 保存该专家在对应 rank 上的物理 ID，
    可供 prefill / decode 路由以及其他布局查询复用，不包含阶段相关策略。
    """
    num_ranks = len(rank_to_logic_expert_ids)
    assert num_ranks > 0
    assert num_logical_experts % num_ranks == 0
    slots_per_rank = len(rank_to_logic_expert_ids[0])
    assert all(len(row) == slots_per_rank for row in rank_to_logic_expert_ids)
    physical_ids_by_expert = _collect_physical_ids_by_logical_expert(rank_to_logic_expert_ids, num_logical_experts)
    logical_to_rank_physical_map = [[-1] * num_ranks for _ in range(num_logical_experts)]
    for expert, physical_ids in enumerate(physical_ids_by_expert):
        assert physical_ids, f"logical expert {expert} has no physical slot"
        for physical_id in physical_ids:
            rank = physical_id // slots_per_rank
            assert (
                logical_to_rank_physical_map[expert][rank] == -1
            ), f"logical expert {expert} has duplicate slots on rank {rank}"
            logical_to_rank_physical_map[expert][rank] = physical_id
    return logical_to_rank_physical_map


def build_minimal_logical_to_physical_map(
    rank_to_logic_expert_ids: LayerPlacement, num_logical_experts: int
) -> list[int]:
    """构建最小副本映射，每个逻辑专家只保留一个执行槽位。

    可用于 prefill 或 decode 中需要控制活跃副本规模的分发；这里的
    “最小”指每个专家只有一个物理副本，不代表通信量的全局最优解。

    输入为单层布局 ``[W, slots_per_rank]``，返回 CPU list ``[E]``：
    ``minimal_physical_map[logical_id] = rank * slots_per_rank + local_slot``。
    按每个 rank ``E // W`` 个专家的目标贪心分配；剩余专家优先填到
    已分配数量较少的可用 rank。物理 ID 始终使用实际布局的槽位。

    每个 rank 按逻辑 ID 升序选择尚未分配的专家。例如布局
    ``[[2, 0, 1], [3, 1, 2]]`` 返回 ``[1, 2, 5, 3]``。
    函数不修改输入，不按当前 source rank 变化，也不优先使用本卡冗余副本。

    固定连续主专家的 decode_cooccurrence 布局直接使用贪心结果。
    候选不足时只填补未分配专家，保留已有 owner，不回溯或重新匹配。
    补选尽量均衡，但不保证严格均分或最小单卡专家数上限。
    """
    assert num_logical_experts > 0, "num_logical_experts must be positive"
    # 副本表同时校验布局：各卡槽位数相同、逻辑 ID 合法、覆盖所有专家，
    # 且单卡不能有同一逻辑专家的两个副本。
    logical_to_rank_physical_map = build_logical_to_rank_physical_map(
        rank_to_logic_expert_ids=rank_to_logic_expert_ids,
        num_logical_experts=num_logical_experts,
    )
    num_ranks = len(rank_to_logic_expert_ids)
    num_experts_per_rank = num_logical_experts // num_ranks
    minimal_physical_map = [-1] * num_logical_experts
    rank_expert_counts = [0] * num_ranks

    for rank, rank_expert_ids in enumerate(rank_to_logic_expert_ids):
        # 前面的 rank 已经选中的专家不再参与选择；排序不改变原始物理槽位。
        candidates = [expert for expert in sorted(rank_expert_ids) if minimal_physical_map[expert] == -1]
        if len(candidates) < num_experts_per_rank:
            break
        for expert in candidates[:num_experts_per_rank]:
            minimal_physical_map[expert] = logical_to_rank_physical_map[expert][rank]
            rank_expert_counts[rank] += 1

    # 只补齐 -1，不改变已选槽位；每次选已分配数量最少的可用 rank，同数量时选较小 rank。
    for expert, physical_id in enumerate(minimal_physical_map):
        if physical_id != -1:
            continue
        available_ranks = [
            rank for rank, candidate_id in enumerate(logical_to_rank_physical_map[expert]) if candidate_id >= 0
        ]
        selected_rank = min(available_ranks, key=lambda rank: (rank_expert_counts[rank], rank))
        minimal_physical_map[expert] = logical_to_rank_physical_map[expert][selected_rank]
        rank_expert_counts[selected_rank] += 1

    assert -1 not in minimal_physical_map, "minimal_physical_map contains unassigned logical experts"
    return minimal_physical_map


def _collect_physical_ids_by_logical_expert(
    rank_to_logic_expert_ids: LayerPlacement,
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
    node_world_size: int,
) -> None:
    """按当前 rank、当前节点、其他节点的优先级稳定排序副本。

    当前 rank 的排序键为 0，同节点其他 rank 为 1，其他节点为 2。同一优先级
    内保持原 physical ID 顺序不变。
    """
    current_node = current_rank // node_world_size

    def locality_priority(physical_expert_id: int) -> int:
        physical_rank = physical_expert_id // num_physical_experts_per_rank
        if physical_rank == current_rank:
            return 0
        if physical_rank // node_world_size == current_node:
            return 1
        return 2

    for physical_expert_ids in physical_ids_by_logical_expert:
        # list.sort 是稳定排序：排序键相同时，physical ID 的原始顺序不变。
        physical_expert_ids.sort(key=locality_priority)


def _build_routing_row(
    physical_expert_ids: list[int],
    num_node_replicas: int,
    num_current_gpu_replicas: int,
    num_routing_slots: int,
) -> list[int]:
    """将一个 logical expert 的候选 physical IDs 打包为固定宽度路由行。

    ``physical_expert_ids`` 已由调用方完成拓扑优先的稳定排序，所以本函数
    不再依赖 ``current_rank``。列表长度就是该 logical expert 的有效物理
    副本数，无需额外传入容易失配的副本数量。
    """
    # 阶段 1：候选列表包含该专家的全部物理副本，其长度就是有效副本数。
    num_global_replicas = len(physical_expert_ids)
    assert 0 < num_global_replicas <= num_routing_slots
    assert 0 <= num_current_gpu_replicas <= num_node_replicas <= num_global_replicas

    # 阶段 2：有效槽位直接保存稳定排序后的候选；固定宽度中未使用的尾部
    # 槽位统一填充 -1。kernel 的副本索引严格小于 num_global_replicas，
    # 因而不会读取 padding。
    num_padding_slots = num_routing_slots - num_global_replicas
    routing_slots = physical_expert_ids + [-1] * num_padding_slots

    # 阶段 3：将三层有效副本计数放在固定头部，后面拼接按拓扑优先级排序的
    # physical IDs 和 -1 padding：
    #
    # [global_count, node_count, current_gpu_count, physical_ids..., -1 padding...]
    return [num_global_replicas, num_node_replicas, num_current_gpu_replicas, *routing_slots]
