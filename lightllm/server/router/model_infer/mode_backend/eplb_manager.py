from enum import Enum
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from lightllm.common.basemodel.basemodel import TpPartBaseModel
from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.eplb_placement import (
    build_logical_to_physical_map,
)
from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.eplb_planner import (
    EPLBPlanner,
    ExpertPlacement,
    GreedyEPLBPlanner,
)
from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.fused_moe_weight import (
    FusedMoeWeight,
)
from lightllm.server.metrics.manager import MetricClient
from lightllm.server.router.model_infer.mode_backend.eplb_plan import EPLBPlanTask
from lightllm.server.router.model_infer.mode_backend.eplb_transfer import (
    EPLBTransferInfo,
    PinnedMemoryEPLBTransfer,
    build_transfer_plan,
)
from lightllm.utils.dist_utils import (
    get_global_rank,
    get_global_world_size,
)
from lightllm.utils.envs_utils import (
    get_eplb_rebalance_gain_threshold,
    get_eplb_step_interval,
)
from lightllm.utils.log_utils import init_logger
from lightllm.utils.shm_port_args import get_shm_port_args

logger = init_logger(__name__)
EPLB_EXPERT_ALIGNMENT = 128
EPLB_MIN_AVERAGE_TOKENS_PER_EXPERT = 256
EPLB_EXPERT_IMBALANCE_RATIO_METRIC = "lightllm_eplb_topk_expert_imbalance_ratio"


class EPLBManagerState(Enum):
    """EPLB 管理器在一次负载均衡循环中的阶段。"""

    COLLECTING = "collecting"
    EVALUATING = "evaluating"
    PLANNING = "planning"
    WAIT_PLAN_FINISH = "wait_plan_finish"
    TRANSFERRING = "transferring"


class EPLBManager:
    """由 :meth:`step` 驱动的 EPLB 状态机。

    状态循环如下：

    ``COLLECTING -> EVALUATING -> PLANNING -> WAIT_PLAN_FINISH -> TRANSFERRING -> COLLECTING``

    当收集的平均专家 token 数不足时，``EVALUATING`` 会回到
    ``COLLECTING``；当规划器认为无需调整布局时，``WAIT_PLAN_FINISH`` 会
    回到 ``COLLECTING``。每次调用 :meth:`step` 最多推进一个状态，布局
    规划和权重传输在后台执行，主推理线程负责评估、轮询和提交结果。
    """

    def __init__(self, model: TpPartBaseModel) -> None:
        weights: List[FusedMoeWeight] = _find_fused_moe_weights(model)
        assert weights, "EPLB requires at least one EP MoE layer"

        # 模型与专家拓扑：初始化后保持不变。
        self._weights: List[FusedMoeWeight] = weights
        self.global_rank: int = get_global_rank()
        self.world_size: int = get_global_world_size()
        assert self.world_size > 1, "EPLB requires more than one rank"
        self._eplb_impls = [weight.fuse_moe_impl for weight in weights]

        first_impl = self._eplb_impls[0]
        self.num_logical_experts: int = first_impl.n_routed_experts
        self.num_redundant_experts_per_rank: int = first_impl.num_redundant_experts_per_rank
        self.num_primary_experts_per_rank: int = self.num_logical_experts // self.world_size

        # 评估调度：steps 只在 COLLECTING 状态递增。
        self.step_interval: int = get_eplb_step_interval()
        self.steps: int = 0

        # 分布式通信：控制面与权重传输使用独立的通信组。
        self.control_group = dist.new_group(list(range(self.world_size)), backend="gloo")
        self.transfer_group = dist.new_group(list(range(self.world_size)), backend="gloo")

        # 专家布局与规划器：直接以各层 impl 中的实际冗余专家槽位为准。
        # 本 rank 的布局索引为 [layer][redundant_slot]。
        local_redundant_expert_ids_by_layer = [
            impl.local_logics_expert_ids_list[self.num_primary_experts_per_rank :] for impl in self._eplb_impls
        ]

        # all_gather 后的布局索引为 [rank][layer][redundant_slot]。
        redundant_expert_ids_by_rank_and_layer: List[List[List[int]]] = [[] for _ in range(self.world_size)]
        dist.all_gather_object(
            redundant_expert_ids_by_rank_and_layer,
            local_redundant_expert_ids_by_layer,
            group=self.control_group,
        )

        # 转置为规划器使用的 [layer][rank][redundant_slot]。
        self.current_placement: ExpertPlacement = [
            [redundant_expert_ids_by_rank_and_layer[rank][layer_index] for rank in range(self.world_size)]
            for layer_index in range(len(weights))
        ]
        self.planner: EPLBPlanner = GreedyEPLBPlanner(
            self.world_size,
            self.num_redundant_experts_per_rank,
            expert_alignment=EPLB_EXPERT_ALIGNMENT,
            rebalance_gain_threshold=get_eplb_rebalance_gain_threshold(),
        )

        self.state = EPLBManagerState.COLLECTING
        self.next_evaluation_step = self.step_interval

        if self.global_rank == 0:
            self.metric_client: MetricClient = MetricClient(get_shm_port_args().metric_port)
            logger.info(
                f"eplb enabled layers={len(weights)} num_logical_experts={self.num_logical_experts} "
                f"num_redundant_experts_per_rank={self.num_redundant_experts_per_rank} "
                f"step_interval={self.step_interval} planner={type(self.planner).__name__}"
            )

    def step(self) -> None:
        """在一个安全的推理边界推进一次状态机。"""
        if self.state is EPLBManagerState.COLLECTING:
            self._step_collecting()
            return

        if self.state is EPLBManagerState.EVALUATING:
            self._step_evaluating()
            return

        if self.state is EPLBManagerState.PLANNING:
            self._step_planning()
            return

        if self.state is EPLBManagerState.WAIT_PLAN_FINISH:
            self._step_wait_plan_finish()
            return

        if self.state is EPLBManagerState.TRANSFERRING:
            self._step_transferring()
            return

        raise RuntimeError(f"unknown EPLB manager state: {self.state!r}")

    # 状态处理：与 step() 的分发顺序保持一致。

    def _step_collecting(self) -> None:
        """记录一个采样步，并在采样窗口结束后进入评估状态。"""
        self.steps += 1
        if self.steps < self.next_evaluation_step:
            return

        self.next_evaluation_step += self.step_interval
        self.state = EPLBManagerState.EVALUATING

    def _step_evaluating(self) -> None:
        """将负载复制到 CPU，并根据全局样本量进入采样或规划状态。"""
        counters = [impl.route_counter for impl in self._eplb_impls]
        if any(counter.ndim != 1 or counter.shape[0] != self.num_logical_experts for counter in counters):
            raise RuntimeError("EPLB route counter shape must be [num_logical_experts]")

        # 将各层累计的路由计数复制到 CPU，后续规划统一使用这份快照。
        local_load = torch.stack([counter.detach().cpu() for counter in counters])

        # 汇集各 rank 的 token 总数，判断当前统计量是否足以进行布局规划。
        token_count_by_rank = [0] * self.world_size
        dist.all_gather_object(
            token_count_by_rank,
            int(local_load.sum().item()),
            group=self.control_group,
        )
        average_tokens_per_expert = sum(token_count_by_rank) / local_load.numel()
        if average_tokens_per_expert < EPLB_MIN_AVERAGE_TOKENS_PER_EXPERT:
            if self.global_rank == 0:
                logger.info(
                    "eplb continue collecting average_tokens_per_expert=%.2f threshold=%s",
                    average_tokens_per_expert,
                    EPLB_MIN_AVERAGE_TOKENS_PER_EXPERT,
                )
            self.state = EPLBManagerState.COLLECTING
            return

        self._local_load = local_load
        self.state = EPLBManagerState.PLANNING

    def _step_planning(self) -> None:
        """汇集全局负载，并由 rank 0 启动异步规划。"""
        local_load = self._local_load
        del self._local_load

        # 一次分配连续的 [rank][layer][logical_expert] 缓冲区，再沿 rank 维
        # 切出 all_gather 所需的输出 tensor。
        gathered_load = torch.empty(
            (self.world_size, *local_load.shape),
            dtype=local_load.dtype,
            device=local_load.device,
        )
        load_by_rank = list(gathered_load.unbind(dim=0))
        dist.all_gather(load_by_rank, local_load, group=self.control_group)
        global_load = gathered_load.sum(dim=0)
        self._publish_expert_load_metric(global_load)

        self.state = EPLBManagerState.WAIT_PLAN_FINISH
        if self.global_rank == 0:
            self._plan_task = EPLBPlanTask(
                self.planner,
                global_load,
                self.current_placement,
            )
            self._plan_task.start()

    def _step_wait_plan_finish(self) -> None:
        """等待 rank 0 完成规划并广播目标专家排布。"""
        placement: Optional[ExpertPlacement] = None
        if self.global_rank == 0 and self._plan_task.is_finished():
            placement = self._plan_task.result
            assert placement is not None

        values = [placement]
        dist.broadcast_object_list(values, src=0, group=self.control_group)
        placement = values[0]
        if placement is None:
            return

        if self.global_rank == 0:
            del self._plan_task

        if placement == self.current_placement:
            if self.global_rank == 0:
                logger.info("eplb skip rearrangement because placement is unchanged")
            self.state = EPLBManagerState.COLLECTING
            return

        self.target_placement: ExpertPlacement = [
            [list(expert_ids) for expert_ids in layer_placement] for layer_placement in placement
        ]
        self.pending_transfer_infos = [
            transfer_info
            for layer_index, (current_layer, target_layer) in enumerate(
                zip(self.current_placement, self.target_placement)
            )
            for transfer_info in build_transfer_plan(
                current_layer,
                target_layer,
                layer_index,
                self.num_logical_experts,
                self.world_size,
            )
        ]
        if not self.pending_transfer_infos:
            raise RuntimeError("planned EPLB rearrangement must contain at least one transfer")
        self.state = EPLBManagerState.TRANSFERRING
        if self.global_rank == 0:
            changed_layer_count = sum(
                current != target for current, target in zip(self.current_placement, self.target_placement)
            )
            logger.info(
                "eplb started steps=%s changed_layer_count=%s changed_slot_count=%s",
                self.steps,
                changed_layer_count,
                len(self.pending_transfer_infos),
            )

    def _step_transferring(self) -> None:
        """启动或轮询一个传输批次；整批完成后再统一提交。"""
        if not hasattr(self, "rebalance_started_at"):
            self.rebalance_started_at = time.time()

        # 没有活动批次时，所有 rank 根据相同的 pending 列表构造下一批任务。
        if not hasattr(self, "active_transfer_batch"):
            transfer_batch = self._pop_next_transfer_batch()

            # 空批次表示公共任务列表已经耗尽，所有 rank 可以同时结束重排。
            if not transfer_batch:
                self.current_placement = self.target_placement
                elapsed = time.time() - self.rebalance_started_at
                del self.pending_transfer_infos
                del self.target_placement
                del self.rebalance_started_at
                self.state = EPLBManagerState.COLLECTING
                if self.global_rank == 0:
                    logger.info("eplb completed wall_time=%.2fs", elapsed)
                return

            self.active_transfer_batch = transfer_batch

            # 一个批次内每个 rank 至多参与一条任务。参与者构造并启动本地传输；
            # 其他 rank 只保存相同的批次信息，后续共同参与状态同步和 commit。
            local_transfer_info = next(
                (
                    transfer_info
                    for transfer_info in transfer_batch
                    if self.global_rank in (transfer_info.source_rank, transfer_info.dest_rank)
                ),
                None,
            )
            if local_transfer_info is not None:
                self.active_transfer = PinnedMemoryEPLBTransfer(
                    self._weights,
                    self.transfer_group,
                    self.global_rank,
                    local_transfer_info,
                )
                self.active_transfer.start()
        else:
            # 已有活动批次时，本 step 只负责轮询；整批完成后才统一提交。
            self._poll_transfer_batch()

    def _pop_next_transfer_batch(self) -> List[EPLBTransferInfo]:
        """从 pending 中取出一组 rank 互不冲突的传输任务。"""
        occupied_ranks = set()
        transfer_batch: List[EPLBTransferInfo] = []
        remaining_transfer_infos: List[EPLBTransferInfo] = []

        # 所有 rank 持有相同的计划列表，并执行相同的贪心扫描，因此会得到完全
        # 相同的批次。每个 rank 在本批次中至多参与一条任务，可以独立启动。
        for transfer_info in self.pending_transfer_infos:
            participant_ranks = {transfer_info.source_rank, transfer_info.dest_rank}
            # 与已选任务没有公共 rank 时，当前任务可以并入本批次。
            if not participant_ranks & occupied_ranks:
                transfer_batch.append(transfer_info)
                occupied_ranks |= participant_ranks
            else:
                remaining_transfer_infos.append(transfer_info)

        # 选中的任务由 active_transfer_batch 持有；未选中的冲突任务保留到下一批。
        self.pending_transfer_infos = remaining_transfer_infos
        return transfer_batch

    def _poll_transfer_batch(self) -> None:
        """等待当前批次全部完成，随后统一提交并释放本地任务。"""
        active_transfer = getattr(self, "active_transfer", None)
        local_state: Optional[Tuple[EPLBTransferInfo, bool]] = None
        if active_transfer is not None:
            local_state = (
                active_transfer.transfer_info,
                active_transfer.is_finished(),
            )
        transfer_states: List[Optional[Tuple[EPLBTransferInfo, bool]]] = [None] * self.world_size
        dist.all_gather_object(transfer_states, local_state, group=self.control_group)

        # 批次内任意任务只要缺少参与方状态，或任一参与方尚未完成，整批都不能
        # commit。下一次 step 会继续轮询同一个批次。
        if not all(state is None or state[1] for state in transfer_states):
            return

        # 所有 rank 使用相同的批次顺序提交，因此全局 placement 和 metadata
        # 始终一致；只有 destination rank 会额外写入实际专家权重。
        from lightllm.server.router.model_infer.infer_batch import g_infer_context

        torch.cuda.current_stream().wait_stream(g_infer_context.get_overlap_stream())
        for transfer_info in self.active_transfer_batch:
            self._commit_transfer(transfer_info)

        if active_transfer is not None:
            del self.active_transfer
        del self.active_transfer_batch

    def _commit_transfer(self, transfer_info: EPLBTransferInfo) -> None:
        """提交一条传输，并发布更新后的路由 metadata。"""
        is_destination_rank = transfer_info.dest_rank == self.global_rank
        if is_destination_rank:
            active_transfer = getattr(self, "active_transfer", None)
            assert (
                active_transfer is not None and active_transfer.transfer_info == transfer_info
            ), "EPLB destination rank has no matching completed transfer"
            for tensor_buffer in active_transfer.tensor_buffers:
                tensor_buffer.live_tensor[transfer_info.dest_local_expert_index].copy_(tensor_buffer.pinned_row)

        layer_index = transfer_info.layer_index
        layer_impl = self._eplb_impls[layer_index]
        redundant_slot_index = transfer_info.dest_local_expert_index - self.num_primary_experts_per_rank
        self.current_placement[layer_index][transfer_info.dest_rank][
            redundant_slot_index
        ] = transfer_info.source_logical_expert_id
        if is_destination_rank:
            layer_impl.local_logics_expert_ids_list[
                transfer_info.dest_local_expert_index
            ] = transfer_info.source_logical_expert_id

        local_expert_ids_by_rank = [
            list(
                range(
                    rank * self.num_primary_experts_per_rank,
                    (rank + 1) * self.num_primary_experts_per_rank,
                )
            )
            + self.current_placement[layer_index][rank]
            for rank in range(self.world_size)
        ]
        logical_to_physical_map = torch.tensor(
            build_logical_to_physical_map(
                local_expert_ids_by_rank,
                self.num_logical_experts,
                current_rank=self.global_rank,
            ),
            dtype=torch.int32,
        )
        layer_impl.logical_to_physical_map.copy_(logical_to_physical_map)

    def _publish_expert_load_metric(self, global_load: torch.Tensor) -> None:
        if self.global_rank != 0:
            return
        self.metric_client.gauge_set(
            EPLB_EXPERT_IMBALANCE_RATIO_METRIC,
            _expert_load_imbalance_ratio(global_load),
        )


def _find_fused_moe_weights(model: TpPartBaseModel) -> List[FusedMoeWeight]:
    weights_by_id: Dict[int, FusedMoeWeight] = {}
    for layer in model.trans_layers_weight:
        for value in getattr(layer, "__dict__", {}).values():
            if isinstance(value, FusedMoeWeight) and value.enable_ep_moe:
                weights_by_id[id(value)] = value
    return sorted(weights_by_id.values(), key=lambda weight: weight.layer_num_)


def _expert_load_imbalance_ratio(global_load: torch.Tensor) -> float:
    """Average each layer's maximum-to-mean logical-expert token ratio."""
    if global_load.ndim != 2:
        raise ValueError("global_load must be [layers, logical_experts]")
    global_load = global_load.to(torch.float64)
    layer_means = global_load.mean(dim=1)
    valid_layers = layer_means > 0
    if not torch.any(valid_layers):
        return 0.0
    ratios = global_load.max(dim=1).values[valid_layers] / layer_means[valid_layers]
    return float(ratios.mean().item())
