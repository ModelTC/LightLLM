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
