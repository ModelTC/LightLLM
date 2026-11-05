"""Decode expert placement based on logical-expert co-occurrence."""

from collections import deque

import torch

from .planner import EPLBPlanner
from .config import validate_decode_layer_placement
from .types import ExpertPlacement, LayerPlacement


class DecodeCooccurrenceEPLBPlanner(EPLBPlanner):
    """固定规范主专家槽，用 decode 共现计数选择冗余专家。

    每个 rank 从自己的规范主专家集合出发，逐个加入共现贡献最高的外部专家；
    每次加入的冗余专家都会参与后续候选评分。
    输入和输出都必须保持规范主专家前缀，只调整末尾的冗余槽。
    目标集合确定后，共同专家继续复用原物理槽位，新选冗余专家填入空槽。
    同一 logical expert 可以复制到多个 rank，但每个 rank 内不能重复，
    冗余槽只需不超过本卡缺少的 logical expert 数，不限制副本的原属 rank。
    这里只规划物理布局，运行时副本选择由独立的 decode 路由算子处理。

    Decode counter 只保存上三角。规划前先聚合所有源 rank，再恢复为对称
    矩阵并清零评分矩阵的主对角线。直接累加共现计数评分，不做逐层归一化。
    相同 score 使用更小的 logical expert ID 打破平局，保证相同输入
    始终得到相同布局。
    """

    def __init__(self, world_size: int, num_redundant_experts_per_rank: int) -> None:
        assert world_size > 1, "world_size must be greater than one"
        assert num_redundant_experts_per_rank > 0, "num_redundant_experts_per_rank must be positive"

        self.world_size = world_size
        self.num_redundant_experts_per_rank = num_redundant_experts_per_rank

    def plan(self, route_statistics: torch.Tensor, current_placement: ExpertPlacement) -> ExpertPlacement:
        """返回 ``[layer][rank][local physical slot]`` 完整目标布局。"""
        self._validate_statistics(route_statistics)

        _, num_layers, num_logical_experts, _ = route_statistics.shape
        self._validate_placement(
            current_placement,
            num_layers=num_layers,
            num_logical_experts=num_logical_experts,
        )

        # [source_rank, layer, expert, expert] -> [layer, expert, expert]
        # counter 的下三角按定义为零，只使用上三角可避免意外重复计数。
        upper_counts = torch.triu(route_statistics.sum(dim=0).to(torch.float64))
        symmetric_counts = upper_counts + upper_counts.transpose(-1, -2)
        # 只考虑不同专家之间的共现；清零新评分矩阵，不修改原始专家负载计数。
        symmetric_counts.diagonal(dim1=-2, dim2=-1).zero_()

        target_placement = [
            self._plan_layer(
                cooccurrence_counts=symmetric_counts[layer_index],
                current_layer=current_placement[layer_index],
            )
            for layer_index in range(num_layers)
        ]
        self._validate_placement(
            target_placement,
            num_layers=num_layers,
            num_logical_experts=num_logical_experts,
        )
        return target_placement

    def _plan_layer(self, cooccurrence_counts: torch.Tensor, current_layer: LayerPlacement) -> LayerPlacement:
        """固定主专家，逐 rank 选择冗余专家并尽量复用当前物理槽。"""
        num_logical_experts = cooccurrence_counts.shape[0]
        num_primary_experts_per_rank = num_logical_experts // self.world_size
        # 输入布局已保证规范主槽；按逻辑 ID 生成同一主专家集合，保持主槽不变。
        primary_experts_by_rank = [
            list(range(rank * num_primary_experts_per_rank, (rank + 1) * num_primary_experts_per_rank))
            for rank in range(self.world_size)
        ]

        selected_by_rank = self._select_redundant_experts(cooccurrence_counts, primary_experts_by_rank)
        return [
            self._reuse_current_slots(
                selected_experts=primary_experts + selected_by_rank[rank],
                current_rank_placement=current_layer[rank],
            )
            for rank, primary_experts in enumerate(primary_experts_by_rank)
        ]

    def _select_redundant_experts(
        self,
        cooccurrence_counts: torch.Tensor,
        primary_experts_by_rank: list[list[int]],
    ) -> list[list[int]]:
        """各 rank 独立扩展专家集合，逐次选取共现贡献最高的外部专家。"""
        selected_by_rank: list[list[int]] = []
        for primary_experts in primary_experts_by_rank:
            # 初始分数是候选与本卡主专家的共现计数之和；主专家不能再次占冗余槽。
            candidate_score = cooccurrence_counts[:, primary_experts].sum(dim=1).to(torch.float64)
            candidate_score[primary_experts] = -torch.inf
            selected_redundant = []
            for _ in range(self.num_redundant_experts_per_rank):
                # argmax 在同分时取第一个位置，即较小 logical expert ID。
                selected_expert = int(candidate_score.argmax().item())
                selected_redundant.append(selected_expert)

                # 新副本加入本卡集合：增量累加它与所有候选的共现计数。
                # 已占用槽位的专家保持 -inf，保证本卡不会重复选择同一专家。
                candidate_score += cooccurrence_counts[:, selected_expert]
                candidate_score[selected_expert] = -torch.inf
            selected_by_rank.append(selected_redundant)
        return selected_by_rank

    def _reuse_current_slots(self, selected_experts: list[int], current_rank_placement: list[int]) -> list[int]:
        """重排完整新旧布局，保留共同专家的原物理槽位，减少无意义迁移。"""
        assert len(selected_experts) == len(current_rank_placement), "slot reuse requires layouts with equal sizes"
        # 新旧布局都是完整物理槽位列表，直接使用旧槽位索引，不需要主槽偏移。
        old_slots = {expert: slot for slot, expert in enumerate(current_rank_placement)}
        new_placement = [None] * len(selected_experts)
        remaining_experts = deque(selected_experts)

        # 按初始队列长度处理一轮：旧专家放回原槽，新专家暂存队尾，保留评分顺序。
        for _ in range(len(remaining_experts)):
            expert = remaining_experts.popleft()
            old_slot = old_slots.get(expert)
            if old_slot is not None:
                new_placement[old_slot] = expert
            else:
                remaining_experts.append(expert)

        # 从左到右填补空槽；队首弹出保留候选顺序，避免 list.pop(0) 搬移元素。
        for slot, expert in enumerate(new_placement):
            if expert is None:
                new_placement[slot] = remaining_experts.popleft()
        return new_placement

    def _validate_statistics(self, route_statistics: torch.Tensor) -> None:
        """校验 decode planner 的原始全局统计 shape。"""
        assert route_statistics.device.type == "cpu", "decode route statistics must be on CPU"
        assert route_statistics.ndim == 4, "decode route statistics must have shape [rank, layer, expert, expert]"
        num_ranks, num_layers, num_experts, last_dim = route_statistics.shape
        assert num_ranks == self.world_size, "decode route statistics rank dimension does not match world_size"
        assert (
            num_layers > 0 and num_experts > 0 and num_experts == last_dim
        ), "decode route statistics must contain square non-empty expert matrices"
        assert num_experts % self.world_size == 0, "logical expert count must be divisible by world_size"
        num_primary = num_experts // self.world_size
        assert (
            self.num_redundant_experts_per_rank <= num_experts - num_primary
        ), "decode redundant experts per rank cannot exceed the number of non-local logical experts"

    def _validate_placement(self, placement: ExpertPlacement, num_layers: int, num_logical_experts: int) -> None:
        """校验完整布局的 shape、覆盖关系、rank 内唯一性和规范主槽。"""
        assert len(placement) == num_layers, "placement layer count does not match route statistics"

        num_primary = num_logical_experts // self.world_size
        num_local_experts = num_primary + self.num_redundant_experts_per_rank
        expected_experts = set(range(num_logical_experts))
        for layer in placement:
            assert len(layer) == self.world_size, "placement rank count does not match world_size"
            for rank_experts in layer:
                assert len(rank_experts) == num_local_experts, "placement contains an invalid number of local experts"
                assert len(set(rank_experts)) == num_local_experts, "placement contains duplicate experts on one rank"
            placed_experts = {expert for rank_experts in layer for expert in rank_experts}
            assert placed_experts == expected_experts, "placement does not cover every logical expert"
            validate_decode_layer_placement(
                layer_placement=layer,
                num_logical_experts=num_logical_experts,
                world_size=self.world_size,
            )
