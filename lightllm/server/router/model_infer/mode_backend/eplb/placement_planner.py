"""使用纯 Python 实现 EPLB 专家布局规划。

规划器有意使用嵌套 list，而不是 Tensor。Tensor 转换仅发生在 manager 的
分布式通信和迁移边界；规划模块不依赖 Tensor，更易于阅读、测试和替换算法。
"""

import heapq
from abc import ABC, abstractmethod
from math import ceil
from typing import List, Tuple


# [layer][logical expert]
LogicalExpertLoad = List[List[float]]
# [layer][rank][local physical expert] -> logical expert
ExpertPlacement = List[List[List[int]]]
# (logical expert, replica count, aligned load per replica)
ExpertReplicaGroup = Tuple[int, int, float]


class EPLBPlanner(ABC):
    """专家布局规划接口。"""

    @abstractmethod
    def plan(
        self,
        logical_expert_load: LogicalExpertLoad,
        current_placement: ExpertPlacement,
    ) -> ExpertPlacement:
        """返回完整的 ``[layer][rank][local physical expert]`` 专家布局。"""


class GreedyEPLBPlanner(EPLBPlanner):
    """使用两阶段启发式算法生成完整的专家布局。

    设计目标
    --------
    对每一层的全局逻辑专家负载进行快速近似均衡，同时满足以下约束：

    * 每个 rank 的物理专家槽位数量必须固定；
    * 每个逻辑专家至少有一个副本；
    * 同一个逻辑专家不能在同一个 rank 上出现两次；
    * 尽量减少最繁忙 rank 的负载，并保持结果确定，方便规划和迁移测试。

    ``num_redundant_experts_per_rank`` 个槽位用于复制专家。这里的“冗余专家”
    是布局中的副本槽位，不代表某些主专家槽位不可移动；当前实现允许所有
    物理槽位重新排列。

    单层规划流程
    ------------
    1. 选择全卡冗余专家：取负载最高的 ``R`` 个逻辑专家，并将它们各放一份
       到所有 rank。它们在每个 rank 上的负载贡献完全相同，因此在后续比较
       rank 之间的相对负载时可以暂时忽略。
    2. 决定额外副本数：全卡冗余专家占用了 ``R * world_size`` 个槽位，剩余
       槽位总数正好是逻辑专家数 ``E``。非冗余专家先各保留一个副本，再将
       多出的 ``R`` 个副本逐次加给当前 ``load / replica_count`` 最高的专家。
       这样拆分后的每个副本负载尽量接近。
    3. 平铺多副本专家：副本数大于 1 的专家按顺序使用循环 rank 游标放置。
       一组副本数最多为 ``world_size``，所以它会落在互不相同的 rank；连续
       使用游标还使第一阶段各 rank 的槽位数最多相差一个。
    4. 放置单副本专家：根据第三步已经形成的 ``rank_load`` 建立最小堆，按
       单副本负载从高到低取专家，每次分给当前负载最低且仍有空槽的 rank。
       此阶段只有一个副本的专家，不存在同专家重复约束；rank 填满后从堆中
       移除。这里负载是第一优先级，剩余槽位数量只用于判断 rank 是否已满。
    5. 复用当前布局：按当前 rank 顺序执行贪心匹配，每次从尚未使用的候选
       rank 中选择共同专家数量最多的一行，先减少跨 rank 的专家迁移；随后
       让共同专家继续占用原物理槽位，再减少 rank 内的权重搬运。候选 rank
       重排和本地槽位复用都不会改变规划负载。

    示例
    ----
    假设 ``E=8``、``world_size=4``、``R=1``、``expert_alignment=1``，逻辑
    专家负载为 ``[40, 12, 9, 8, 7, 5, 4, 3]``。总物理槽位数为
    ``E + R * world_size = 12``，所以每个 rank 必须恰好放置三个专家。

    1. 选择全卡冗余专家

       专家 0 的负载 40 最高，因此每个 rank 都先放置专家 0。四个副本各
       分担 ``40 / 4 = 10`` 的负载：

       ``placement = [[0], [0], [0], [0]]``

       此时每个 rank 的公共负载都是 10、剩余槽位都是 2。公共负载不会影响
       rank 之间的大小关系，所以后续平衡过程只记录非冗余专家的负载。

    2. 计算非冗余专家的副本数

       去掉专家 0 后，专家 1 到 7 先各保留一个副本，只能占用 7 个槽位；但
       当前共有 8 个剩余槽位，因此还需要增加一个副本。专家 1 的当前单副本
       负载 12 最高，所以将其拆成两个负载为 6 的副本。最终专家组为：

       ``[(expert=1, copies=2, load=6),``
       `` (expert=2..7, copies=1, load=9, 8, 7, 5, 4, 3)]``

    3. 第一阶段平铺多副本专家

       循环游标从 rank 0 开始，将专家 1 的两个副本依次放到 rank 0、1：

       ``placement = [[0, 1], [0, 1], [0], [0]]``
       ``rank_load = [6, 6, 0, 0]``
       ``remaining_slots = [1, 1, 2, 2]``

    4. 第二阶段分配单副本专家

       专家 2 到 7 已按负载从高到低排列。每次从最小堆中取当前负载最低的
       未满 rank；负载相同时使用 rank ID 打破平局：

       * 专家 2，负载 9：放到 rank 2，负载变为 ``[6, 6, 9, 0]``；
       * 专家 3，负载 8：放到 rank 3，负载变为 ``[6, 6, 9, 8]``；
       * 专家 4，负载 7：放到 rank 0，负载变为 ``[13, 6, 9, 8]``，rank 0 填满；
       * 专家 5，负载 5：放到 rank 1，负载变为 ``[13, 11, 9, 8]``，rank 1 填满；
       * 专家 6，负载 4：放到 rank 3，负载变为 ``[13, 11, 9, 12]``，rank 3 填满；
       * 专家 7，负载 3：放到 rank 2，负载变为 ``[13, 11, 12, 12]``，rank 2 填满。

       最终候选布局为：

       ``rank 0: [0, 1, 4]``
       ``rank 1: [0, 1, 5]``
       ``rank 2: [0, 2, 7]``
       ``rank 3: [0, 3, 6]``

       将专家 0 的公共负载 10 加回来后，完整 rank 负载为
       ``[23, 21, 22, 22]``，平均负载为 22，最大负载为 23。

    5. 复用当前布局

       上述 rank 编号只是负载分组结果。规划器会依次处理当前 rank 0 到 3，
       每次从尚未匹配的候选行中选择共同专家最多的一行；匹配完成后，再让
       共同专家尽量保留原物理槽位。两次排序都不改变负载，完成后直接返回
       新布局。

    合法性和确定性
    --------------
    * 专家覆盖：全卡冗余专家和非冗余专家组来自互斥集合，并且两者合起来
      包含所有逻辑专家，所以不会遗漏任何专家。
    * 本地去重：一个多副本专家最多有 ``world_size`` 个副本，循环游标在一组
      副本分配完成前不会第二次经过同一 rank；单副本专家只放置一次。因此
      同一逻辑专家不会在一个 rank 上出现两次。
    * 槽位守恒：全卡冗余专家放置完成后，剩余副本总数严格等于剩余槽位总数。
      第一阶段连续平铺使各 rank 的已用槽位数最多相差一个；第二阶段只从尚有
      空位的 rank 中选择，并在 rank 填满后将其移出最小堆，最终所有槽位恰好
      填满。
    * 结果确定：专家组排序和最小堆比较最终都使用专家 ID 或 rank ID 打破
      平局，因此相同输入始终得到相同布局。
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
        """逐层规划专家布局，再组合成完整的多层布局。"""
        # 先把外部输入转换成规划器内部统一使用的 Python 数值类型，并一次性
        # 校验所有层的形状和布局约束。后续每层规划之间没有共享的可变状态。
        load = [[float(value) for value in layer] for layer in logical_expert_load]
        current = [[[int(expert) for expert in rank] for rank in layer] for layer in current_placement]
        self._validate_inputs(load, current)

        # 每层只依赖自己的逻辑专家负载和当前布局。先完成单层规划，再将结果
        # 按原 layer 顺序组合，避免多层候选和负载数据交叉索引。
        return [self._plan_layer(layer_load, current_layer) for layer_load, current_layer in zip(load, current)]

    def _plan_layer(
        self,
        logical_load: List[float],
        current_placement: List[List[int]],
    ) -> List[List[int]]:
        """完成单层副本分配、rank 排布和物理槽位复用。"""
        # 阶段 1：选出最热的 R 个专家，并为每个 rank 固定预留它们的副本。
        redundant_experts = self._select_redundant_experts(logical_load)

        # 阶段 2：只在非冗余专家中增加副本，数量恰好填满所有剩余槽位。
        remaining_expert_groups = self._build_remaining_expert_groups(logical_load, redundant_experts)

        # 阶段 3：每个 rank 先放入相同的冗余专家，再通过 rank 优先队列
        # 分配其余专家。同一专家的一组副本会一次性放到不同 rank。
        candidate_placement = self._distribute_remaining_experts(redundant_experts, remaining_expert_groups)

        # 阶段 4：先将候选行贪心匹配到最相似的当前 rank，再复用原物理槽位，
        # 依次减少跨 rank 迁移和 rank 内部的槽位搬运。
        return self._reuse_current_slots(candidate_placement, current_placement)

    def _select_redundant_experts(self, logical_load: List[float]) -> List[int]:
        """选择需要在所有 rank 上固定放置的最热专家。"""
        return sorted(range(len(logical_load)), key=lambda expert: (-logical_load[expert], expert))[
            : self.num_redundant_experts_per_rank
        ]

    def _build_remaining_expert_groups(
        self,
        logical_load: List[float],
        redundant_experts: List[int],
    ) -> List[ExpertReplicaGroup]:
        """确定非冗余专家的副本数，并按安全的分配顺序组成专家组。"""
        redundant_expert_set = set(redundant_experts)
        remaining_experts = [expert for expert in range(len(logical_load)) if expert not in redundant_expert_set]
        replica_counts = {expert: 1 for expert in remaining_experts}

        # 每个 rank 的 R 个槽位已由全卡冗余专家占据。此时剩余槽位总数为
        # logical_expert_count，而未分配专家只有 logical_expert_count-R 个，
        # 所以还需在非冗余专家中增加 R 个副本。
        for _ in range(self.num_redundant_experts_per_rank):
            expert = min(
                (expert for expert in remaining_experts if replica_counts[expert] < self.world_size),
                key=lambda expert: (-logical_load[expert] / replica_counts[expert], expert),
            )
            replica_counts[expert] += 1

        # 分配器按专家组工作，而不是把同一专家拆成多个独立元素。这样分配
        # 一组副本时可以暂时取出多个不同 rank，从结构上避免本地重复专家。
        expert_groups = []
        for expert in remaining_experts:
            replica_count = replica_counts[expert]
            load_per_replica = logical_load[expert] / replica_count
            aligned_load_per_replica = ceil(load_per_replica / self.expert_alignment) * self.expert_alignment
            expert_groups.append((expert, replica_count, aligned_load_per_replica))

        # 多副本专家需要在第一阶段先完成平铺，因此排在单副本专家之前。
        # 副本数相同时优先处理单副本负载较高的专家，最后用专家 ID 打破平局。
        expert_groups.sort(key=lambda group: (-group[1], -group[2], group[0]))
        return expert_groups

    def _distribute_remaining_experts(
        self,
        redundant_experts: List[int],
        expert_groups: List[ExpertReplicaGroup],
    ) -> List[List[int]]:
        """先平铺多副本专家，再按当前 rank 负载分配单副本专家。"""
        placement = [list(redundant_experts) for _ in range(self.world_size)]

        total_replica_count = sum(replica_count for _, replica_count, _ in expert_groups)
        assert total_replica_count % self.world_size == 0
        remaining_slots_per_rank = total_replica_count // self.world_size
        remaining_slots = [remaining_slots_per_rank] * self.world_size
        rank_load = [0.0] * self.world_size

        replicated_expert_groups = [group for group in expert_groups if group[1] > 1]
        single_expert_groups = [group for group in expert_groups if group[1] == 1]

        # 阶段 1：用同一个循环游标依次平铺所有多副本专家。每个专家最多有
        # world_size 个副本，所以一组副本在游标绕回起点之前已经分配完毕，
        # 同一 rank 不会出现该专家的两个副本。连续使用同一个游标还会让
        # 各 rank 在第一阶段获得的槽位数最多相差一个，不会提前填满某个 rank。
        next_rank = 0
        for expert, replica_count, load_per_replica in replicated_expert_groups:
            for _ in range(replica_count):
                assert remaining_slots[next_rank] > 0
                placement[next_rank].append(expert)
                remaining_slots[next_rank] -= 1
                rank_load[next_rank] += load_per_replica
                next_rank = (next_rank + 1) % self.world_size

        # 阶段 2：多副本专家的位置固定后，再把尚有空位的 rank 按当前负载
        # 放入最小堆。这里负载是第一优先级，剩余槽位数不再参与排序；每次
        # 都把当前最热的单副本专家交给最轻的未满 rank。
        # 全卡冗余专家对每个 rank 的贡献相同，因此无需计入 rank_load。
        rank_queue = [(rank_load[rank], rank) for rank in range(self.world_size) if remaining_slots[rank] > 0]
        heapq.heapify(rank_queue)

        for expert, replica_count, load_per_replica in single_expert_groups:
            assert replica_count == 1
            assert rank_queue, "not enough rank slots to place remaining experts"

            current_load, rank = heapq.heappop(rank_queue)
            placement[rank].append(expert)
            remaining_slots[rank] -= 1

            # remaining_slots == 0 表示该 rank 已经刚好填满，不再放回队列。
            if remaining_slots[rank] > 0:
                heapq.heappush(rank_queue, (current_load + load_per_replica, rank))

        assert not rank_queue
        assert all(slots == 0 for slots in remaining_slots)

        return placement

    def _reuse_current_slots(
        self,
        candidate_placement: List[List[int]],
        current_placement: List[List[int]],
    ) -> List[List[int]]:
        """贪心匹配候选 rank，并让共同专家尽量复用当前物理槽位。"""
        # 步骤 1：准备所有尚未匹配的候选行。
        #
        # 候选布局中的 rank 编号只是负载规划阶段产生的临时编号。任意交换
        # 两个候选行都不会改变每行的专家组合和整体负载，因此可以重新排列
        # 候选行，使其尽量贴近当前运行布局。这里同时缓存专家集合，后续可以
        # 直接用集合交集计算两个 rank 之间的相似度。
        unmatched_candidates = [
            (candidate_rank, candidate_experts, set(candidate_experts))
            for candidate_rank, candidate_experts in enumerate(candidate_placement)
        ]
        placement = []

        # 步骤 2：按当前 rank 0 -> N-1 的顺序贪心匹配候选行。
        #
        # 相似度定义为两个 rank 共同持有的专家数量。共同专家越多，需要跨
        # rank 传输的专家权重就越少。一个候选行被选中后立即从待选列表移除，
        # 从而建立当前 rank 和候选行之间的一一对应关系。
        for current_experts in current_placement:
            current_expert_set = set(current_experts)

            # max() 首先选择共同专家数量最多的候选行。相似度相同时，负的
            # candidate_rank 让原候选 rank ID 更小的行优先，保证结果确定。
            best_candidate_index = max(
                range(len(unmatched_candidates)),
                key=lambda index: (
                    len(current_expert_set & unmatched_candidates[index][2]),
                    -unmatched_candidates[index][0],
                ),
            )
            _, selected_experts, selected_expert_set = unmatched_candidates.pop(best_candidate_index)

            # 步骤 3：在已经匹配的 rank 内复用当前物理槽位。
            #
            # new_experts 只包含候选行新引入的专家，并保持候选行中的原始顺序。
            # 它的数量必然等于当前行中需要被替换的专家数量。
            new_experts = [expert for expert in selected_experts if expert not in current_expert_set]
            new_expert_index = 0
            rank_placement = []

            # 依次检查当前物理槽位：如果槽位中的专家仍被候选行选中，就原地
            # 保留；否则用下一个新专家填充。这样只有真正变化的槽位需要搬运
            # 权重，共同专家不会因为候选行内部顺序不同而发生无意义移动。
            for current_expert in current_experts:
                if current_expert in selected_expert_set:
                    rank_placement.append(current_expert)
                    continue

                rank_placement.append(new_experts[new_expert_index])
                new_expert_index += 1

            # 所有需要替换的槽位都应恰好消费一个新专家。
            assert new_expert_index == len(new_experts)
            placement.append(rank_placement)

        # 每个当前 rank 都必须匹配且只匹配一个候选行。
        assert not unmatched_candidates
        return placement

    def _validate_inputs(
        self,
        logical_expert_load: LogicalExpertLoad,
        placement: ExpertPlacement,
    ) -> None:
        """拒绝会导致逐层规划静默截断的输入。"""
        if not logical_expert_load:
            raise ValueError("logical_expert_load must contain at least one layer")
        if len(placement) != len(logical_expert_load):
            raise ValueError("load and placement must have the same number of layers")
