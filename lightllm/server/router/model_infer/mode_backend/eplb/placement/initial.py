"""Construct the deterministic expert placement used during model loading."""

from .types import LayerPlacement


def build_initial_local_expert_ids(
    num_logical_experts: int,
    num_ranks: int,
    num_redundant_experts_per_rank: int,
) -> LayerPlacement:
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
