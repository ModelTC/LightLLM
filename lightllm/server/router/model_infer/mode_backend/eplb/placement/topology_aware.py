"""固定主专家并感知节点拓扑的 EPLB 布局规划器。"""

from typing import Optional

import torch

from .planner import EPLBPlanner
from .types import ExpertPlacement


class TopologyAwareEPLBPlanner(EPLBPlanner):
    """根据源节点流量规划冗余专家，并最小化各层关键 rank 的计算负载。

    与会重新排列全部物理槽位的 ``GlobalBalanceEPLBPlanner`` 不同，本规划器固定
    每个 rank 的规范主专家，只修改末尾的冗余专家槽位。规划过程保留原始
    sample 维和流量来源节点，使用与 ``current_node_first`` 路由一致的负载
    模型：源节点存在目标专家副本时只使用节点内副本，否则回退到全局副本。

    算法包含五个主要阶段：

    1. 把 ``[rank, layer, sample, expert]`` 负载聚合为
       ``[sample, layer, source_node, expert]``；
    2. 从当前完整布局中提取冗余槽位，固定主专家不参与候选规划；
    3. 各层独立填充冗余槽位，每次选择使关键 rank 计算量最小的专家副本；
    4. 只接受关键负载下降的层，并要求模型级收益达到配置阈值；
    5. 让保留在同一 rank 的副本继续占用原物理槽，再拼回完整布局。

    负载评估会对每个 sample、每个物理专家分别执行 ``expert_alignment``
    向上对齐，从而避免先合并多个 prefill 批次后低估实际计算量。
    """

    def __init__(
        self,
        world_size: int,
        num_redundant_experts_per_rank: int,
        node_world_size: int,
        expert_alignment: int = 128,
        rebalance_gain_threshold: float = 0.05,
    ) -> None:
        assert world_size > 1, "world_size must be greater than one"
        assert num_redundant_experts_per_rank > 0, "num_redundant_experts_per_rank must be positive"
        assert expert_alignment > 0, "expert_alignment must be positive"

        assert (
            node_world_size > 0 and world_size % node_world_size == 0
        ), "node_world_size must be positive and divide world_size"
        assert 0.0 <= rebalance_gain_threshold <= 1.0

        self.world_size = world_size
        self.node_world_size = node_world_size
        self.num_redundant_experts_per_rank = num_redundant_experts_per_rank
        self.expert_alignment = expert_alignment
        self.rebalance_gain_threshold = rebalance_gain_threshold

    def plan(
        self,
        logical_expert_load_samples: torch.Tensor,
        current_placement: ExpertPlacement,
    ) -> ExpertPlacement:
        """返回 ``[layer][rank][local physical slot]`` 形式的完整目标布局。"""
        assert logical_expert_load_samples.device.type == "cpu"
        assert logical_expert_load_samples.ndim == 4
        assert logical_expert_load_samples.shape[0] == self.world_size
        assert logical_expert_load_samples.shape[1] > 0

        num_layers = logical_expert_load_samples.shape[1]
        num_logical_experts = logical_expert_load_samples.shape[3]
        self._validate_complete_placement(
            current_placement,
            num_layers=num_layers,
            num_logical_experts=num_logical_experts,
        )

        # 步骤 1：保留 sample 和源节点信息。
        #
        # 输入 shape： [rank, layer, sample, logical_expert]
        # 输出 shape： [sample, layer, source_node, logical_expert]
        #
        # 同一节点内的 rank 使用相同的本地副本集合，因此先在节点内部聚合；
        # 不聚合不同节点，也不聚合 sample，避免丢失拓扑和逐批次对齐信息。
        source_node_load = self._aggregate_load_by_source_node(logical_expert_load_samples).to(torch.float64)

        # 步骤 2：把当前完整布局转换为算法内部使用的冗余布局。
        #
        # 返回 None 表示当前布局的主专家前缀已经被其他 planner 改动，无法
        # 使用固定主专家模型做公平的 before/after 比较。此时仍会生成新的
        # 规范布局，但本轮无法与旧布局比较收益，也无法复用旧冗余槽位。
        current_redundant_placement = self._try_extract_redundant_placement(
            current_placement,
            num_logical_experts=num_logical_experts,
        )

        # 步骤 3：以关键 rank 负载为目标逐个填充冗余槽位。
        candidate_redundant_placement = self._plan_redundant_experts(source_node_load)

        if current_redundant_placement is not None:
            # 步骤 4：先逐层排除退化候选，再检查整个模型的收益比例。
            # 候选未达到阈值时直接返回当前冗余布局，不触发专家迁移。
            candidate_redundant_placement = self._select_improving_placement(
                source_node_load,
                current_placement=current_redundant_placement,
                candidate_placement=candidate_redundant_placement,
            )

            # 步骤 5：planner 只决定“哪个 rank 保存哪些专家”，冗余槽位本身
            # 可以互换。这里让保留下来的专家继续占用原槽，只用空出的槽位
            # 接收新专家，避免候选顺序变化造成没有意义的 rank 内复制。
            candidate_redundant_placement = self._reuse_current_redundant_slots(
                candidate_redundant_placement,
                current_redundant_placement,
            )

        # 状态机、路由表和传输规划器使用完整布局，因此最后重新补上每个
        # rank 的规范主专家前缀，并在返回前校验 planner 生成结果。
        target_placement = self._build_complete_placement(
            candidate_redundant_placement,
            num_logical_experts=num_logical_experts,
        )
        self._validate_complete_placement(
            target_placement,
            num_layers=num_layers,
            num_logical_experts=num_logical_experts,
        )
        return target_placement

    def _aggregate_load_by_source_node(
        self,
        load_samples: torch.Tensor,
    ) -> torch.Tensor:
        """聚合同一节点内的 rank，同时保留 sample 和节点维度。"""
        _, num_layers, num_samples, num_logical_experts = load_samples.shape
        num_nodes = self.world_size // self.node_world_size

        # global rank 按节点连续编号。先拆成
        # [node, rank_in_node, layer, sample, expert]，只消去 rank_in_node。
        load_by_node = load_samples.reshape(
            num_nodes,
            self.node_world_size,
            num_layers,
            num_samples,
            num_logical_experts,
        ).sum(dim=1)

        # 调整为后续负载模型统一使用的
        # [sample, layer, source_node, expert]。
        return load_by_node.permute(2, 1, 0, 3).contiguous()

    def _try_extract_redundant_placement(
        self,
        placement: ExpertPlacement,
        num_logical_experts: int,
    ) -> Optional[torch.Tensor]:
        """校验主专家前缀，并提取 ``[layer, rank, redundant_slot]`` 布局。

        ``global_balance`` 模式或历史配置可能移动了主专家槽位。这样的完整布局在
        运行时仍然合法，但不符合本 planner 的固定主专家模型，因此返回
        ``None``，通知调用方跳过旧布局收益比较和冗余槽位复用。
        """
        num_primary_experts_per_rank = num_logical_experts // self.world_size

        # 每个 rank 的规范主专家是一个连续区间：
        # rank r -> [r * primary_count, (r + 1) * primary_count)。
        for layer_placement in placement:
            for rank, rank_placement in enumerate(layer_placement):
                primary_start = rank * num_primary_experts_per_rank
                expected_primary = list(range(primary_start, primary_start + num_primary_experts_per_rank))
                if rank_placement[:num_primary_experts_per_rank] != expected_primary:
                    return None

        # 主专家前缀已经确认规范，直接截取每行末尾的冗余槽位。
        return torch.tensor(
            [
                [rank_placement[-self.num_redundant_experts_per_rank :] for rank_placement in layer_placement]
                for layer_placement in placement
            ],
            dtype=torch.int64,
        )

    def _validate_complete_placement(
        self,
        placement: ExpertPlacement,
        num_layers: int,
        num_logical_experts: int,
    ) -> None:
        """校验完整布局的基本 shape 和 logical expert 覆盖范围。"""
        assert num_logical_experts % self.world_size == 0
        assert len(placement) == num_layers

        num_primary_experts_per_rank = num_logical_experts // self.world_size
        num_local_experts = num_primary_experts_per_rank + self.num_redundant_experts_per_rank
        expected_experts = set(range(num_logical_experts))

        for layer_placement in placement:
            assert len(layer_placement) == self.world_size
            assert all(len(rank_placement) == num_local_experts for rank_placement in layer_placement)
            placed_experts = {expert for rank_placement in layer_placement for expert in rank_placement}
            assert placed_experts == expected_experts

    def _plan_redundant_experts(
        self,
        source_node_load: torch.Tensor,
    ) -> torch.Tensor:
        """按层独立规划 ``[layer, rank, redundant_slot]`` 冗余布局。"""
        num_nodes = source_node_load.shape[2]
        local_rank_mask = self._build_local_rank_mask(num_nodes)

        # 各层没有共享的规划状态。显式逐层处理可以让单层算法只操作
        # [sample, source_node, expert] 等三维数据，同时把候选临时内存限制在
        # 单层规模。layer 顺序固定，因此结果仍然具有确定性。
        return torch.stack(
            [
                self._plan_one_layer(
                    source_node_load[:, layer_index],
                    local_rank_mask=local_rank_mask,
                )
                for layer_index in range(source_node_load.shape[1])
            ]
        )

    def _plan_one_layer(
        self,
        source_node_load: torch.Tensor,
        local_rank_mask: torch.Tensor,
    ) -> torch.Tensor:
        """规划一层的 ``[rank, redundant_slot]`` 冗余专家布局。"""
        num_logical_experts = source_node_load.shape[2]
        num_primary_experts_per_rank = num_logical_experts // self.world_size
        num_redundant_slots = self.world_size * self.num_redundant_experts_per_rank

        redundant_placement = torch.full(
            (self.world_size, self.num_redundant_experts_per_rank),
            -1,
            dtype=torch.int64,
        )

        # expert_locations: [logical_expert, rank]。初始状态只包含连续划分的
        # 规范主专家；后续每确定一个冗余副本，就原地提交到该占用矩阵。
        expert_locations = torch.zeros(
            (num_logical_experts, self.world_size),
            dtype=torch.bool,
        )
        expert_ids = torch.arange(num_logical_experts, dtype=torch.int64)
        primary_ranks = expert_ids // num_primary_experts_per_rank
        expert_locations[expert_ids, primary_ranks] = True

        # expert_rank_load: [sample, logical_expert, rank]；rank_load 再沿
        # logical expert 维累加为 [sample, rank]。
        expert_rank_load = self._estimate_one_layer_expert_rank_load(
            source_node_load,
            expert_locations,
            local_rank_mask=local_rank_mask,
        )
        rank_load = expert_rank_load.sum(dim=1)
        remaining_slots = [self.num_redundant_experts_per_rank] * self.world_size

        for _ in range(num_redundant_slots):
            # 先选择累计负载最低且仍有冗余槽位的 rank。expert_locations 已经
            # 包含主副本，因此该 rank 上值为 False 的 expert 就是全部合法候选。
            target_rank = self._select_target_rank(rank_load, remaining_slots)
            legal_experts = ~expert_locations[:, target_rank]
            assert torch.any(legal_experts)

            # 同时模拟把每个 logical expert 放入目标 rank。矩阵的每一行只
            # 影响对应 expert 自己的路由比例，因此可以一次得到所有候选的
            # 新负载贡献，而不必逐 expert 执行模拟。
            candidate_locations = expert_locations.clone()
            candidate_locations[:, target_rank] = True
            candidate_expert_rank_load = self._estimate_one_layer_expert_rank_load(
                source_node_load,
                candidate_locations,
                local_rank_mask=local_rank_mask,
            )

            # 对候选 expert e，只把 e 的旧负载贡献替换为新增副本后的贡献。
            # candidate_rank_load: [sample, candidate_expert, rank]。
            candidate_rank_load = rank_load[:, None, :] - expert_rank_load + candidate_expert_rank_load
            candidate_critical_load = self._critical_load(candidate_rank_load)
            candidate_critical_load.masked_fill_(~legal_experts, torch.inf)
            selected_expert = int(candidate_critical_load.argmin().item())
            assert torch.isfinite(candidate_critical_load[selected_expert])

            target_slot = self.num_redundant_experts_per_rank - remaining_slots[target_rank]
            redundant_placement[target_rank, target_slot] = selected_expert

            # 只提交被选 expert 的负载增量、占用关系和槽位状态，未被选择的
            # 候选模拟结果在本轮结束后直接丢弃。
            selected_new_load = candidate_expert_rank_load[:, selected_expert]
            selected_old_load = expert_rank_load[:, selected_expert]
            rank_load += selected_new_load - selected_old_load
            expert_rank_load[:, selected_expert] = selected_new_load
            expert_locations[selected_expert, target_rank] = True
            remaining_slots[target_rank] -= 1

        assert torch.all(redundant_placement >= 0)
        return redundant_placement

    def _select_target_rank(
        self,
        rank_load: torch.Tensor,
        remaining_slots: list[int],
    ) -> int:
        """选择累计负载最低且仍有冗余槽位的 rank。"""
        rank_order = torch.argsort(rank_load.sum(dim=0), stable=True)
        for rank in rank_order.tolist():
            if remaining_slots[rank] > 0:
                return rank
        assert False, "topology-aware planner found no available redundant slot"

    def _select_improving_placement(
        self,
        source_node_load: torch.Tensor,
        current_placement: torch.Tensor,
        candidate_placement: torch.Tensor,
    ) -> torch.Tensor:
        """逐层过滤退化候选，并应用模型级最小收益门槛。"""
        local_rank_mask = self._build_local_rank_mask(source_node_load.shape[2])
        current_rank_load = self._estimate_rank_load(
            source_node_load,
            current_placement,
            local_rank_mask=local_rank_mask,
        )
        candidate_rank_load = self._estimate_rank_load(
            source_node_load,
            candidate_placement,
            local_rank_mask=local_rank_mask,
        )

        # 先对每个 sample 取最繁忙 rank，再沿 sample 求和。每层只有候选值
        # 严格小于当前值时才允许替换，避免用其他层的收益掩盖本层退化。
        current_critical_by_layer = self._critical_load(current_rank_load)
        candidate_critical_by_layer = self._critical_load(candidate_rank_load)
        improved_layers = candidate_critical_by_layer < current_critical_by_layer
        selected_critical_by_layer = torch.where(
            improved_layers,
            candidate_critical_by_layer,
            current_critical_by_layer,
        )

        # 再计算所有已选择层合在一起的模型级收益。只有关键计算量下降比例
        # 达到阈值，才值得支付专家权重迁移成本；否则整次规划保持不变。
        current_model_critical = current_critical_by_layer.sum()
        selected_model_critical = selected_critical_by_layer.sum()
        rebalance_gain = (current_model_critical - selected_model_critical) / current_model_critical.clamp_min(1.0)
        if rebalance_gain.item() < self.rebalance_gain_threshold:
            return current_placement.clone()

        selected_placement = current_placement.clone()
        selected_placement[improved_layers] = candidate_placement[improved_layers]
        return selected_placement

    def _critical_load(self, rank_load: torch.Tensor) -> torch.Tensor:
        """每个 sample 取最繁忙 rank，再沿 sample 累加。"""
        return rank_load.amax(dim=-1).sum(dim=0)

    def _estimate_rank_load(
        self,
        source_node_load: torch.Tensor,
        redundant_placement: torch.Tensor,
        local_rank_mask: torch.Tensor,
    ) -> torch.Tensor:
        """估算冗余布局对应的 ``[sample, layer, rank]`` 计算负载。"""
        num_logical_experts = source_node_load.shape[-1]
        locations = self._expert_locations(
            redundant_placement,
            num_logical_experts=num_logical_experts,
        )
        expert_rank_load = self._estimate_expert_rank_load(
            source_node_load,
            locations,
            local_rank_mask=local_rank_mask,
        )
        return expert_rank_load.sum(dim=2)

    def _build_local_rank_mask(self, num_nodes: int) -> torch.Tensor:
        """构造 ``[source_node, rank]`` 同节点关系矩阵。"""
        assert num_nodes == self.world_size // self.node_world_size
        source_nodes = torch.arange(num_nodes, dtype=torch.int64)
        rank_nodes = torch.arange(self.world_size, dtype=torch.int64) // self.node_world_size
        return source_nodes[:, None] == rank_nodes[None, :]

    def _expert_locations(
        self,
        redundant_placement: torch.Tensor,
        num_logical_experts: int,
    ) -> torch.Tensor:
        """构造包含主副本的 ``[layer, logical_expert, rank]`` 占用矩阵。"""
        num_layers = redundant_placement.shape[0]
        num_primary_experts_per_rank = num_logical_experts // self.world_size
        locations = torch.zeros(
            (num_layers, num_logical_experts, self.world_size),
            dtype=torch.bool,
        )

        # 主专家固定连续划分，每个 expert 恰好有一个主副本。
        expert_ids = torch.arange(num_logical_experts, dtype=torch.int64)
        primary_ranks = expert_ids // num_primary_experts_per_rank
        locations[:, expert_ids, primary_ranks] = True

        # 冗余布局按 rank-major、slot-minor 展平。规划中的 -1 表示该槽尚未
        # 填充，需要通过 valid mask 排除，不能作为 expert ID 参与索引。
        redundant_ids = redundant_placement.reshape(num_layers, -1)
        redundant_ranks = torch.arange(
            self.world_size,
            dtype=torch.int64,
        ).repeat_interleave(self.num_redundant_experts_per_rank)
        valid = redundant_ids >= 0
        if torch.any(valid):
            layer_indices = torch.arange(
                num_layers,
                dtype=torch.int64,
            ).unsqueeze(1)
            layer_indices = layer_indices.expand_as(redundant_ids)
            rank_indices = redundant_ranks[None, :].expand_as(redundant_ids)
            locations[
                layer_indices[valid],
                redundant_ids[valid],
                rank_indices[valid],
            ] = True
        return locations

    def _estimate_expert_rank_load(
        self,
        source_node_load: torch.Tensor,
        locations: torch.Tensor,
        local_rank_mask: torch.Tensor,
    ) -> torch.Tensor:
        """计算对齐后的 ``[sample, layer, expert, rank]`` 负载贡献。"""
        route_fraction = self._route_fraction(
            locations,
            local_rank_mask=local_rank_mask,
        )
        physical_load = torch.einsum(
            "slne,lner->sler",
            source_node_load,
            route_fraction,
        )
        return self._align_expert_load(physical_load)

    def _estimate_one_layer_expert_rank_load(
        self,
        source_node_load: torch.Tensor,
        locations: torch.Tensor,
        local_rank_mask: torch.Tensor,
    ) -> torch.Tensor:
        """计算单层 ``[sample, expert, rank]`` 的对齐后负载贡献。"""
        route_fraction = self._route_fraction(
            locations,
            local_rank_mask=local_rank_mask,
        )
        physical_load = torch.einsum(
            "sne,ner->ser",
            source_node_load,
            route_fraction,
        )
        return self._align_expert_load(physical_load)

    def _route_fraction(
        self,
        locations: torch.Tensor,
        local_rank_mask: torch.Tensor,
    ) -> torch.Tensor:
        """根据节点本地优先规则计算每个副本承接的路由比例。"""
        # locations 可以是 [layer, expert, rank] 或 [expert, rank]。
        # 在 rank 前插入 source_node 维后，与 [source_node, 1, rank]
        # 的拓扑 mask 广播，分别得到 [layer, node, expert, rank] 或
        # [node, expert, rank]。
        all_copies = locations.unsqueeze(-3)
        local_copies = all_copies & local_rank_mask.unsqueeze(-2)

        # 源节点存在副本时只使用节点内副本，否则回退到全部全局副本。
        selected_copies = torch.where(
            local_copies.any(dim=-1, keepdim=True),
            local_copies,
            all_copies,
        )
        return selected_copies.to(torch.float64) / selected_copies.sum(
            dim=-1,
            keepdim=True,
        )

    def _align_expert_load(self, physical_load: torch.Tensor) -> torch.Tensor:
        """按 DeepEP expert alignment 向上对齐物理专家负载。"""
        return torch.ceil(physical_load / self.expert_alignment) * self.expert_alignment

    def _reuse_current_redundant_slots(
        self,
        candidate_placement: torch.Tensor,
        current_placement: torch.Tensor,
    ) -> torch.Tensor:
        """保留仍在同一 rank 的专家槽位，只用释放槽位接收新专家。"""
        aligned_placement = torch.empty_like(candidate_placement)
        num_layers = candidate_placement.shape[0]

        for layer_index in range(num_layers):
            for rank in range(self.world_size):
                candidate_row = candidate_placement[layer_index, rank].tolist()
                current_row = current_placement[layer_index, rank].tolist()
                candidate_set = set(candidate_row)
                current_set = set(current_row)

                # 当前布局中不再需要的 expert 对应可以覆盖的物理槽位。
                freed_slots = [slot for slot, expert in enumerate(current_row) if expert not in candidate_set]

                # 候选布局中新出现的 expert 按 planner 产生的稳定顺序写入。
                new_experts = [expert for expert in candidate_row if expert not in current_set]
                assert len(freed_slots) == len(new_experts)

                aligned_row = current_row[:]
                for slot, expert in zip(freed_slots, new_experts):
                    aligned_row[slot] = expert
                aligned_placement[layer_index, rank] = torch.tensor(
                    aligned_row,
                    dtype=torch.int64,
                )

        return aligned_placement

    def _build_complete_placement(
        self,
        redundant_placement: torch.Tensor,
        num_logical_experts: int,
    ) -> ExpertPlacement:
        """在冗余布局前补上规范主专家，生成状态机使用的完整布局。"""
        num_primary_experts_per_rank = num_logical_experts // self.world_size
        complete_placement: ExpertPlacement = []

        for layer_redundant_placement in redundant_placement.tolist():
            layer_placement = []
            for rank, redundant_experts in enumerate(layer_redundant_placement):
                primary_start = rank * num_primary_experts_per_rank
                primary_experts = list(
                    range(
                        primary_start,
                        primary_start + num_primary_experts_per_rank,
                    )
                )
                layer_placement.append(primary_experts + redundant_experts)
            complete_placement.append(layer_placement)

        return complete_placement
