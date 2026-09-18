from enum import Enum
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from lightllm.common.basemodel.basemodel import TpPartBaseModel
from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.eplb_placement import (
    build_logical_to_physical_maps_for_layers,
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

        self.state = EPLBManagerState.WAIT_PLAN_FINISH
        if self.global_rank == 0:
            self._plan_task = EPLBPlanTask(
                self.planner,
                global_load,
                self.current_placement,
            )
            self._plan_task.start()

    def _step_wait_plan_finish(self) -> None:
        """等待 rank 0 完成规划并广播结果。"""
        result: Optional[Dict[str, Any]] = None
        if self.global_rank == 0 and self._plan_task.is_finished():
            result = self._plan_task.result
            assert result is not None

        values = [result]
        dist.broadcast_object_list(values, src=0, group=self.control_group)
        result = values[0]
        if result is None:
            return

        if self.global_rank == 0:
            del self._plan_task

        self._publish_expert_load_metric(result)
        if result["kind"] != "planned":
            if self.global_rank == 0:
                logger.info("eplb skip rearrangement kind=%s", result["kind"])
            self.state = EPLBManagerState.COLLECTING
            return

        result["metadata"], result["transfer_infos"] = self._build_rebalance_data(result)
        pending_transfer_infos = list(result["transfer_infos"])
        if not pending_transfer_infos:
            raise RuntimeError("planned EPLB rearrangement must contain at least one transfer")

        self.target_placement: ExpertPlacement = [
            [list(expert_ids) for expert_ids in layer_placement] for layer_placement in result["placement"]
        ]
        self.target_metadata = result["metadata"]
        self.pending_transfer_infos = pending_transfer_infos
        self.completed_layer_transfers: List[PinnedMemoryEPLBTransfer] = []
        self.rebalance_started_at = time.time()
        self.state = EPLBManagerState.TRANSFERRING
        self._start_next_transfer()
        if self.global_rank == 0:
            logger.info(
                "eplb started steps=%s max_before=%.4f max_after=%.4f "
                "p95_before=%.4f p95_after=%.4f rebalance_gain=%.4f "
                "changed_layer_count=%s changed_slot_count=%s",
                self.steps,
                result["before"]["max"],
                result["after"]["max"],
                result["before"]["p95"],
                result["after"]["p95"],
                result["rebalance_gain"],
                result["changed_layer_count"],
                len(self.pending_transfer_infos),
            )

    def _step_transferring(self) -> None:
        """推进当前传输，并在一层完成后原子地发布该层。"""
        if not self._active_transfer_finished_on_all_ranks():
            return

        completed_info = self._complete_active_transfer()
        if self._next_transfer_is_in_layer(completed_info.layer_index):
            self._start_next_transfer()
            return

        self._synchronize_and_commit_layer(completed_info.layer_index)
        if self.pending_transfer_infos:
            self._start_next_transfer()
            return

        elapsed = self._complete_rebalance()
        self.state = EPLBManagerState.COLLECTING
        if self.global_rank == 0:
            logger.info(
                "eplb completed wall_time=%.2fs",
                elapsed,
            )

    def _publish_expert_load_metric(self, result: Dict[str, Any]) -> None:
        if self.global_rank != 0:
            return
        self.metric_client.gauge_set(EPLB_EXPERT_IMBALANCE_RATIO_METRIC, result["expert_imbalance_ratio"])

    def _build_rebalance_data(self, result: Dict[str, Any]) -> Tuple[Dict[int, torch.Tensor], List[EPLBTransferInfo]]:
        metadata_by_layer: Dict[int, torch.Tensor] = {}
        planned_transfers: List[EPLBTransferInfo] = []
        changed_layer_indices: List[int] = [
            layer_index for layer_index, changed in enumerate(result["changed_layers"]) if changed
        ]

        num_primary_experts_per_rank = self.num_logical_experts // self.world_size
        local_expert_ids_by_rank_and_layer: List[List[List[int]]] = []
        for layer_index in changed_layer_indices:
            local_expert_ids_by_rank_and_layer.append(
                [
                    list(
                        range(
                            rank * num_primary_experts_per_rank,
                            (rank + 1) * num_primary_experts_per_rank,
                        )
                    )
                    + result["placement"][layer_index][rank]
                    for rank in range(self.world_size)
                ]
            )
        logical_to_physical_maps = torch.tensor(
            build_logical_to_physical_maps_for_layers(
                local_expert_ids_by_rank_and_layer,
                self.num_logical_experts,
                current_rank=self.global_rank,
            ),
            dtype=torch.int32,
        )
        for changed_layer_offset, layer_index in enumerate(changed_layer_indices):
            current_layer_placement = self.current_placement[layer_index]
            target_layer_placement = result["placement"][layer_index]
            metadata_by_layer[layer_index] = logical_to_physical_maps[changed_layer_offset]
            planned_transfers.extend(
                build_transfer_plan(
                    current_layer_placement,
                    target_layer_placement,
                    layer_index,
                    self.num_logical_experts,
                    self.world_size,
                )
            )
        return metadata_by_layer, planned_transfers

    # 传输阶段内部实现。

    def _active_transfer_finished_on_all_ranks(self) -> bool:
        """仅当所有 rank 都完成当前传输时返回 ``True``。"""
        finished_by_rank = [False] * self.world_size
        dist.all_gather_object(
            finished_by_rank,
            self.active_transfer.is_finished(),
            group=self.control_group,
        )
        return all(finished_by_rank)

    def _complete_active_transfer(self) -> EPLBTransferInfo:
        """将当前传输从待处理队列移动到本层的已完成列表。"""
        assert self.pending_transfer_infos

        expected_transfer_info: EPLBTransferInfo = self.pending_transfer_infos[0]
        if self.active_transfer.transfer_info != expected_transfer_info:
            raise RuntimeError("EPLB completed transfer does not match the expected transfer info")

        self.completed_layer_transfers.append(self.active_transfer)
        self.pending_transfer_infos.pop(0)
        return expected_transfer_info

    def _next_transfer_is_in_layer(self, layer_index: int) -> bool:
        """判断下一个待处理传输是否仍属于当前层。"""
        return bool(self.pending_transfer_infos and self.pending_transfer_infos[0].layer_index == layer_index)

    def _start_next_transfer(self) -> None:
        transfer_info: EPLBTransferInfo = self.pending_transfer_infos[0]
        self.active_transfer = PinnedMemoryEPLBTransfer(
            self._weights,
            self.transfer_group,
            self.global_rank,
            transfer_info,
        )
        self.active_transfer.start()

    def _synchronize_and_commit_layer(self, layer_index: int) -> None:
        """等待旧权重使用完毕，然后发布一层的新权重和 metadata。"""
        from lightllm.server.router.model_infer.infer_batch import g_infer_context

        torch.cuda.current_stream().wait_stream(g_infer_context.get_overlap_stream())
        self._commit_transferred_layer(layer_index)
        self.completed_layer_transfers.clear()

    def _complete_rebalance(self) -> float:
        self.current_placement = self.target_placement
        elapsed = time.time() - self.rebalance_started_at
        del self.pending_transfer_infos
        del self.completed_layer_transfers
        del self.active_transfer
        del self.target_placement
        del self.target_metadata
        del self.rebalance_started_at
        return elapsed

    def _commit_transferred_layer(self, layer_index: int) -> None:
        """在主推理线程中同步发布一层权重和路由 metadata。"""
        target_redundant_expert_ids: List[int] = self.target_placement[layer_index][self.global_rank]
        for transfer in self.completed_layer_transfers:
            transfer_info: EPLBTransferInfo = transfer.transfer_info
            if transfer_info.dest_rank != self.global_rank:
                continue
            for tensor_buffer in transfer.tensor_buffers:
                tensor_buffer.live_tensor[transfer_info.dest_local_expert_index].copy_(tensor_buffer.pinned_row)

        local_expert_ids: List[int] = self._eplb_impls[layer_index].local_logics_expert_ids_list
        local_expert_ids[self.num_primary_experts_per_rank :] = target_redundant_expert_ids
        self._commit_layer_metadata(layer_index)

    def _commit_layer_metadata(self, layer_index: int) -> None:
        self._eplb_impls[layer_index].logical_to_physical_map.copy_(self.target_metadata[layer_index])


def _find_fused_moe_weights(model: TpPartBaseModel) -> List[FusedMoeWeight]:
    weights_by_id: Dict[int, FusedMoeWeight] = {}
    for layer in model.trans_layers_weight:
        for value in getattr(layer, "__dict__", {}).values():
            if isinstance(value, FusedMoeWeight) and value.enable_ep_moe:
                weights_by_id[id(value)] = value
    return sorted(weights_by_id.values(), key=lambda weight: weight.layer_num_)
