"""使用纯 Python 实现 EPLB 冗余专家布局规划。

规划器有意使用嵌套 list，而不是 Tensor。Tensor 转换仅发生在 manager 的
分布式通信和迁移边界；规划模块不依赖 Tensor，更易于阅读、测试和替换算法。
"""

from abc import ABC, abstractmethod
from math import ceil
from typing import List


# [layer][logical expert]
LogicalExpertLoad = List[List[float]]
# [layer][rank][local physical expert] -> logical expert
ExpertPlacement = List[List[List[int]]]
# [layer][rank]
RankLoad = List[List[float]]


class EPLBPlanner(ABC):
    """冗余专家布局规划接口。"""

    @abstractmethod
    def plan(
        self,
        logical_expert_load: LogicalExpertLoad,
        current_placement: ExpertPlacement,
    ) -> ExpertPlacement:
        """返回完整的 ``[layer][rank][local physical expert]`` 专家布局。"""


class GreedyEPLBPlanner(EPLBPlanner):
    """逐个填充冗余槽位，尽量降低最繁忙 rank 的负载。

    主专家始终固定不动。每轮枚举所有合法的 ``(rank, logical_expert)`` 组合，
    选择加入副本后最大 rank 负载最小的候选，直至填满全部冗余槽位。
    新增副本只会改变对应逻辑专家的负载分摊，因此可以直接根据当前布局评估
    每个候选，不再需要单独分配副本数量或执行回溯搜索。
    """

    def __init__(
        self,
        world_size: int,
        num_redundant_experts_per_rank: int,
        *,
        expert_alignment: int = 1,
    ):
        if world_size <= 1:
            raise ValueError("world_size must be greater than one")
        if num_redundant_experts_per_rank <= 0:
            raise ValueError("num_redundant_experts_per_rank must be positive")
        if expert_alignment <= 0:
            raise ValueError("expert_alignment must be positive")
        self.world_size = world_size
        self.num_redundant_experts_per_rank = num_redundant_experts_per_rank
        self.expert_alignment = expert_alignment

    def plan(
        self,
        logical_expert_load: LogicalExpertLoad,
        current_placement: ExpertPlacement,
    ) -> ExpertPlacement:
        """根据全局逻辑专家负载生成新的完整布局。

        ``logical_expert_load`` 的形状为 ``[layer][logical_expert]``，保存所有
        rank 汇总后的负载。``current_placement`` 的形状为
        ``[layer][rank][local_physical_expert]``；每个 rank 的固定主专家在前，
        可迁移冗余专家在后。返回布局与当前布局形状相同。
        """
        load = [[float(value) for value in layer] for layer in logical_expert_load]
        current = [[[int(expert) for expert in rank] for rank in layer] for layer in current_placement]
        self._validate_inputs(load, current)
        before_rank_load = self.estimate_rank_load(load, current)

        candidates = [self._plan_layer(layer_load, current_layer) for layer_load, current_layer in zip(load, current)]
        candidate_rank_load = self.estimate_rank_load(load, candidates)
        placement = []
        for layer, candidate in enumerate(candidates):
            # 每层独立决定是否采用候选布局。只有最大 rank 负载严格下降时
            # 才迁移，避免相同负载下的无收益调整，也使零负载层保持原布局。
            before = max(before_rank_load[layer])
            after = max(candidate_rank_load[layer])
            improved = candidate != current[layer] and after < before
            placement.append(candidate if improved else current[layer])
        return placement

    def estimate_rank_load(
        self,
        logical_expert_load: LogicalExpertLoad,
        placement: ExpertPlacement,
    ) -> RankLoad:
        """估算每层、每个 rank 上经过对齐后的物理专家工作量。"""
        load = [[float(value) for value in layer] for layer in logical_expert_load]
        normalized_placement = [[[int(expert) for expert in rank] for rank in layer] for layer in placement]
        num_logical_experts = self._validate_inputs(load, normalized_placement)
        estimated = []
        for layer_load, layer_placement in zip(load, normalized_placement):
            locations = self._expert_locations(layer_placement, num_logical_experts)
            rank_load = [0.0] * self.world_size
            for expert, expert_locations in enumerate(locations):
                for rank, value in enumerate(self._rank_load_for_expert(layer_load[expert], expert_locations)):
                    rank_load[rank] += value
            estimated.append(rank_load)
        return estimated

    def _plan_layer(
        self,
        logical_load: List[float],
        current_placement: List[List[int]],
    ) -> List[List[int]]:
        num_logical_experts = len(logical_load)
        num_primary_experts_per_rank = num_logical_experts // self.world_size
        current_redundant_placement = [row[num_primary_experts_per_rank:] for row in current_placement]

        # 从不可变的主专家布局开始。只要目标 rank 尚未持有该专家副本，
        # 对应的 (rank, expert) 组合就是合法候选。
        redundant_placement: List[List[int]] = [[] for _ in range(self.world_size)]
        locations = [{expert // num_primary_experts_per_rank} for expert in range(num_logical_experts)]
        rank_load_by_expert = [
            self._rank_load_for_expert(logical_load[expert], locations[expert]) for expert in range(num_logical_experts)
        ]
        rank_load = [
            sum(rank_load_by_expert[expert][rank] for expert in range(num_logical_experts))
            for rank in range(self.world_size)
        ]

        total_redundant_experts = self.world_size * self.num_redundant_experts_per_rank
        for _ in range(total_redundant_experts):
            best_score = None
            best_rank = None
            best_expert = None
            best_rank_load = None
            best_expert_rank_load = None
            for rank in range(self.world_size):
                if len(redundant_placement[rank]) == self.num_redundant_experts_per_rank:
                    continue

                for expert in range(num_logical_experts):
                    if rank in locations[expert]:
                        continue

                    next_locations = locations[expert] | {rank}
                    next_expert_rank_load = self._rank_load_for_expert(logical_load[expert], next_locations)
                    next_rank_load = [
                        load - rank_load_by_expert[expert][target_rank] + next_expert_rank_load[target_rank]
                        for target_rank, load in enumerate(rank_load)
                    ]

                    # 首先最小化最大 rank 负载；负载相同时依次选择总对齐工作量
                    # 更小、能保留现有本地副本的候选。最后使用 rank/expert ID
                    # 打破平局，保证相同输入始终得到相同结果。
                    score = (
                        max(next_rank_load),
                        sum(next_rank_load),
                        expert not in current_redundant_placement[rank],
                        rank,
                        expert,
                    )
                    if best_score is None or score < best_score:
                        best_score = score
                        best_rank = rank
                        best_expert = expert
                        best_rank_load = next_rank_load
                        best_expert_rank_load = next_expert_rank_load

            # 输入校验已保证每个 rank 都有足够多且互不重复的非本地主专家，
            # 因此所有冗余槽位一定可以填满。
            assert best_rank is not None and best_expert is not None
            assert best_rank_load is not None and best_expert_rank_load is not None
            redundant_placement[best_rank].append(best_expert)
            locations[best_expert].add(best_rank)
            rank_load = best_rank_load
            rank_load_by_expert[best_expert] = best_expert_rank_load

        # 布局质量只取决于选中了哪些专家，与它们在本地冗余槽中的顺序无关。
        # 已经选中的现有专家继续使用原槽位，只有新增专家才填入剩余槽位；
        # 这样无需干扰上面的负载均衡循环，也能尽量减少权重传输。
        for rank, selected_experts in enumerate(redundant_placement):
            selected_set = set(selected_experts)
            new_experts = iter(expert for expert in selected_experts if expert not in current_redundant_placement[rank])
            redundant_placement[rank] = [
                expert if expert in selected_set else next(new_experts) for expert in current_redundant_placement[rank]
            ]

        return [
            list(
                range(
                    rank * num_primary_experts_per_rank,
                    (rank + 1) * num_primary_experts_per_rank,
                )
            )
            + redundant_experts
            for rank, redundant_experts in enumerate(redundant_placement)
        ]

    def _rank_load_for_expert(
        self,
        logical_expert_load: float,
        locations: set,
    ) -> List[float]:
        """返回该专家在各 rank 上经过对齐后的负载贡献。"""
        physical_expert_load = logical_expert_load / len(locations)
        aligned_load = ceil(physical_expert_load / self.expert_alignment) * self.expert_alignment
        result = [0.0] * self.world_size
        for rank in locations:
            result[rank] = aligned_load
        return result

    def _expert_locations(
        self,
        placement: List[List[int]],
        num_logical_experts: int,
    ) -> List[set]:
        """反转单层布局，并拒绝同一 rank 上的重复专家副本。"""
        locations = [set() for _ in range(num_logical_experts)]
        for rank, row in enumerate(placement):
            for expert in row:
                if rank in locations[expert]:
                    raise ValueError(f"logical expert {expert} appears twice on rank {rank}")
                locations[expert].add(rank)
        return locations

    def _validate_inputs(
        self,
        logical_expert_load: LogicalExpertLoad,
        placement: ExpertPlacement,
    ) -> int:
        """校验完整布局约束，并返回逻辑专家数量。"""
        if not logical_expert_load:
            raise ValueError("logical_expert_load must contain at least one layer")
        if len(placement) != len(logical_expert_load):
            raise ValueError("load and placement must have the same number of layers")
        num_logical_experts = len(logical_expert_load[0])
        if num_logical_experts == 0 or num_logical_experts % self.world_size:
            raise ValueError("logical expert count must be positive and divisible by world_size")
        if self.num_redundant_experts_per_rank > num_logical_experts - num_logical_experts // self.world_size:
            raise ValueError("too many redundant slots to avoid local or duplicate replicas")
        num_primary_experts_per_rank = num_logical_experts // self.world_size
        num_local_experts_per_rank = num_primary_experts_per_rank + self.num_redundant_experts_per_rank

        for layer_load, layer_placement in zip(logical_expert_load, placement):
            if len(layer_load) != num_logical_experts:
                raise ValueError("each load layer must contain every logical expert")
            if any(value < 0 for value in layer_load):
                raise ValueError("logical expert load must be non-negative")
            if len(layer_placement) != self.world_size or any(
                len(rank) != num_local_experts_per_rank for rank in layer_placement
            ):
                raise ValueError("each placement layer must be [world_size][local_experts]")
            if any(expert < 0 or expert >= num_logical_experts for rank in layer_placement for expert in rank):
                raise ValueError("placement contains an invalid logical expert")
            for rank, local_expert_ids in enumerate(layer_placement):
                expected_primary_expert_ids = list(
                    range(
                        rank * num_primary_experts_per_rank,
                        (rank + 1) * num_primary_experts_per_rank,
                    )
                )
                if local_expert_ids[:num_primary_experts_per_rank] != expected_primary_expert_ids:
                    raise ValueError("placement primary experts do not match their owning rank")
            self._expert_locations(layer_placement, num_logical_experts)
        return num_logical_experts
