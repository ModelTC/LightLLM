from enum import Enum
import time
from typing import List, Optional

import torch
import torch.distributed as dist

from lightllm.common.basemodel.basemodel import TpPartBaseModel
from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.fused_moe_weight import (
    FusedMoeWeight,
)
from lightllm.server.metrics.manager import MetricClient
from lightllm.utils.dist_utils import (
    get_global_rank,
    get_global_world_size,
    get_node_world_size,
)
from lightllm.utils.device_utils import is_sm100_gpu
from lightllm.utils.envs_utils import (
    get_eplb_step_interval,
    get_eplb_transfer_layer_parallelism,
)
from lightllm.utils.log_utils import init_logger
from lightllm.utils.shm_port_args import get_shm_port_args

from . import metrics as eplb_metrics
from .async_expert_transfer import (
    EPLBTransferInfo,
    PinnedMemoryEPLBTransfer,
)
from .async_load_gather_task import EPLBLoadGatherTask
from .async_placement_plan_task import EPLBPlanTask
from .async_transfer_planner import EPLBTransferPlanner
from .placement import (
    EPLBPlanner,
    ExpertPlacement,
    build_logical_to_physical_map,
    create_eplb_planner,
    save_placement_config,
)

logger = init_logger(__name__)
EPLB_EXPERT_ALIGNMENT = 128
EPLB_MIN_AVERAGE_TOKENS_PER_EXPERT = 128


class EPLBManagerState(Enum):
    """EPLB 管理器在一次负载均衡循环中的阶段。"""

    COLLECTING = "collecting"
    EVALUATING = "evaluating"
    WAIT_LOAD_GATHER_FINISHED = "wait_load_gather_finished"
    PLAN_PLACEMENT = "plan_placement"
    WAIT_PLAN_PLACEMENT_FINISHED = "wait_plan_placement_finished"
    PLAN_TRANSFER = "plan_transfer"
    WAIT_PLAN_TRANSFER_FINISHED = "wait_plan_transfer_finished"
    TRANSFERRING = "transferring"


class EPLBManager:
    """由 :meth:`step` 驱动的 EPLB 状态机。

    每次调用 :meth:`step` 最多处理一个状态。主路径及各状态的职责如下::

        [COLLECTING]
          prefill kernel 将各层 logical expert 负载写入 24 行环形样本；
          manager 只记录采样 step，等待下一个评估周期。
                |
                | 评估周期到达
                v
        [EVALUATING]
          将本地环形样本完整复制到 CPU 并上报负载指标；在独立 Gloo
          通信组中启动逐样本负载的后台 all-gather。rank 0 在通信前判断
          本地样本量；样本不足或达到重排次数上限时不启动通信。
                |
                v
        [WAIT_LOAD_GATHER_FINISHED]
          等待所有 rank 完成负载汇集，保留 planner 需要的
          rank 和 sample 原始维度。
                |
                | 样本充足且仍允许重排
                v
        [PLAN_PLACEMENT]
          rank 0 使用完整的全局专家负载启动后台布局规划任务。
                |
                v
        [WAIT_PLAN_PLACEMENT_FINISHED]
          轮询 rank 0 的规划任务，并向所有 rank 广播目标布局。
                |
                | 目标布局发生变化
                v
        [PLAN_TRANSFER]
          每个 rank 根据相同的当前/目标布局启动后台传输规划任务。
                |
                v
        [WAIT_PLAN_TRANSFER_FINISHED]
          等待所有 rank 生成一致的、按依赖关系分批的传输任务。
                |
                v
        [TRANSFERRING]
          分批启动并轮询后台权重传输；整批完成后，主推理线程在安全
          边界统一提交权重和路由 metadata。全部批次完成后发布新布局、
          清空 prefill 路由样本并回到 COLLECTING。

    以下分支会提前回到 ``COLLECTING``::

        EVALUATING
          |-- 平均 token 数不足 --------> 保留环形窗口，继续滚动采样
          `-- 已达到重排次数上限 ------> 清空样本，仅周期性上报指标

        WAIT_PLAN_PLACEMENT_FINISHED
          `-- 目标布局与当前布局相同 ---> 保留环形窗口，等待下次评估

    布局规划、传输规划和权重传输在后台执行；主推理线程只负责创建任务、
    轮询状态，以及在安全边界提交已经完成的结果。
    """

    def __init__(
        self,
        model: TpPartBaseModel,
        max_rebalance_count: int = 1,
        config_path: Optional[str] = None,
        plan_mode: str = "topology_aware",
        run_mode: str = "prefill",
    ) -> None:
        # SM100 FP4 Mega-MoE 会将在线专家权重转换为独立的 kernel 布局，并使用源 tensor 的 data_ptr
        # 作为 key 缓存这些转换后的副本。EPLB 通过原地 copy_ 替换专家行，只改变权重内容而不会改变
        # data_ptr，因此重平衡后 Mega-MoE 仍会读取旧的转换权重。在 EPLB 能够失效或更新该缓存前，
        # 暂不支持 SM100。
        assert not is_sm100_gpu(), "EPLB does not support SM100"

        weights: List[FusedMoeWeight] = _find_fused_moe_weights(model)
        assert weights, "EPLB requires at least one EP MoE layer"
        assert max_rebalance_count >= -1
        assert run_mode in ("prefill", "decode")

        # 模型与专家拓扑：初始化后保持不变。
        self._weights: List[FusedMoeWeight] = weights
        self.config_path = config_path
        self.global_rank: int = get_global_rank()
        self.world_size: int = get_global_world_size()
        assert self.world_size > 1, "EPLB requires more than one rank"
        self.node_world_size: int = get_node_world_size()
        self.layer_indexes = [weight.layer_num_ for weight in weights]
        self._eplb_impls = [weight.fuse_moe_impl for weight in weights]

        first_impl = self._eplb_impls[0]
        self.num_logical_experts: int = first_impl.n_routed_experts
        self.num_redundant_experts_per_rank: int = first_impl.num_redundant_experts_per_rank
        # run_mode 决定 EPLB 面向哪类推理负载进行优化。当前先把模式作为
        # manager 的稳定输入；prefill 与 decode 的采样结构可在此基础上分别演进。
        self.run_mode: str = run_mode
        self.plan_mode: str = plan_mode
        self.planner: EPLBPlanner = create_eplb_planner(
            self.plan_mode,
            self.world_size,
            self.num_redundant_experts_per_rank,
            expert_alignment=EPLB_EXPERT_ALIGNMENT,
            node_world_size=self.node_world_size,
        )

        # 评估调度：steps 只在 COLLECTING 状态递增。prefill 路由样本从当前
        # 布局生效时开始写入，并在固定容量内保留最近的采样窗口。
        self.step_interval: int = get_eplb_step_interval()
        self.transfer_layer_parallelism: int = get_eplb_transfer_layer_parallelism()
        self.steps: int = 0
        self.max_rebalance_count: int = max_rebalance_count
        self.completed_rebalance_count: int = 0

        # 一次重排周期中的短期状态。统一初始化为 None，避免各状态
        # 通过 hasattr()/del 隐式定义 EPLBManager 的属性结构。
        self._load_gather_task: Optional[EPLBLoadGatherTask] = None
        self._pending_plan_load_samples: Optional[torch.Tensor] = None
        self._metric_sample_load: Optional[torch.Tensor] = None
        self._plan_task: Optional[EPLBPlanTask] = None
        self.target_placement: Optional[ExpertPlacement] = None
        self._transfer_planner: Optional[EPLBTransferPlanner] = None
        self.pending_transfer_batches: Optional[List[List[EPLBTransferInfo]]] = None
        self.rebalance_started_at: Optional[float] = None
        self.active_transfer_batch: Optional[List[EPLBTransferInfo]] = None
        self.active_transfers: Optional[List[PinnedMemoryEPLBTransfer]] = None

        # 分布式通信：后台负载汇集、主线程控制面和后台权重传输分别使用
        # 独立的 Gloo 通信组。负载 all-gather 可能跨越多个 manager step，
        # 不能与主线程中按 step 排序的控制 collective 共用同一个 group。
        self.load_gather_group = dist.new_group(list(range(self.world_size)), backend="gloo")
        self.control_group = dist.new_group(list(range(self.world_size)), backend="gloo")
        self.transfer_group = dist.new_group(list(range(self.world_size)), backend="gloo")

        # 每层布局都保存完整的本地专家列表，索引为
        # [layer][local_expert]。global_balance 可以重新分配全部物理槽；topology_aware
        # 固定规范主专家前缀，只重新分配末尾的冗余槽。
        local_expert_ids_by_layer = [list(impl.local_logics_expert_ids_list) for impl in self._eplb_impls]

        # all_gather 后的布局索引为 [rank][layer][local_expert]。
        expert_ids_by_rank_and_layer: List[List[List[int]]] = [[] for _ in range(self.world_size)]
        dist.all_gather_object(
            expert_ids_by_rank_and_layer,
            local_expert_ids_by_layer,
            group=self.control_group,
        )

        # 转置为全局统一使用的 [layer][rank][local_expert]。
        self.current_placement: ExpertPlacement = [
            [expert_ids_by_rank_and_layer[rank][layer_index] for rank in range(self.world_size)]
            for layer_index in range(len(weights))
        ]

        self.state = EPLBManagerState.COLLECTING
        self.next_evaluation_step = self.step_interval
        self._clear_prefill_route_samples()

        if self.global_rank == 0:
            self.metric_client: MetricClient = MetricClient(get_shm_port_args().metric_port)
            logger.info(
                f"eplb enabled layers={len(weights)} num_logical_experts={self.num_logical_experts} "
                f"num_redundant_experts_per_rank={self.num_redundant_experts_per_rank} "
                f"step_interval={self.step_interval} max_rebalance_count={self.max_rebalance_count} "
                f"transfer_layer_parallelism={self.transfer_layer_parallelism} "
                f"run_mode={self.run_mode} plan_mode={self.plan_mode} planner={type(self.planner).__name__}"
            )

    def step(self) -> None:
        """在一个安全的推理边界推进一次状态机。"""
        if self.state is EPLBManagerState.COLLECTING:
            self._step_collecting()
            return

        if self.state is EPLBManagerState.EVALUATING:
            self._step_evaluating()
            return

        if self.state is EPLBManagerState.WAIT_LOAD_GATHER_FINISHED:
            self._step_wait_load_gather_finished()
            return

        if self.state is EPLBManagerState.PLAN_PLACEMENT:
            self._step_plan_placement()
            return

        if self.state is EPLBManagerState.WAIT_PLAN_PLACEMENT_FINISHED:
            self._step_wait_plan_placement_finished()
            return

        if self.state is EPLBManagerState.PLAN_TRANSFER:
            self._step_plan_transfer()
            return

        if self.state is EPLBManagerState.WAIT_PLAN_TRANSFER_FINISHED:
            self._step_wait_plan_transfer_finished()
            return

        if self.state is EPLBManagerState.TRANSFERRING:
            self._step_transferring()
            return

        raise RuntimeError(f"unknown EPLB manager state: {self.state!r}")

    # 状态处理：与 step() 的分发顺序保持一致。

    def _step_collecting(self) -> None:
        """记录一个采样步，并在当前评估周期结束后进入评估状态。"""
        self.steps += 1
        if self.steps < self.next_evaluation_step:
            return
        else:
            self.next_evaluation_step += self.step_interval
            self.state = EPLBManagerState.EVALUATING

    def _step_evaluating(self) -> None:
        """快照本地逐样本负载，并在独立通信组中启动后台汇集。"""
        counters = [impl.prefill_route_counter for impl in self._eplb_impls]
        if any(counter.ndim != 2 or counter.shape[1] != self.num_logical_experts for counter in counters):
            raise RuntimeError("EPLB prefill route counter shape must be [sample_capacity, num_logical_experts]")
        if len({counter.shape[0] for counter in counters}) != 1:
            raise RuntimeError("EPLB prefill route counter capacities must match across layers")

        # 一次性堆叠各层环形样本并复制到 CPU，保留完整的
        # [layer, sample, logical_expert] 维度。后续后台 all-gather 会继续保留
        # rank 和 sample 维；通信完成后将原始四维快照交给 planner。
        # 此处先不清零 GPU 样本：如果样本不足或无需迁移，下一周期会继续
        # 滚动覆盖最旧行；达到重排上限或成功切换到新布局后才重置窗口。本轮
        # 异步规划使用独立的 CPU 快照，不会与后续的 atomic add 竞争。
        local_load_samples = torch.stack(counters).detach().cpu()
        if self.global_rank == 0:
            local_load = local_load_samples.sum(dim=1)
            eplb_metrics.publish_expert_load_metrics(
                metric_client=self.metric_client,
                expert_load=local_load,
            )

        # 达到重排次数上限后仍保留周期性负载上报，但不再执行后续的跨 rank
        # 通信和布局规划。清空本轮样本，使下一次指标对应新的采样窗口。
        reached_rebalance_limit = (
            self.max_rebalance_count != -1 and self.completed_rebalance_count >= self.max_rebalance_count
        )
        if reached_rebalance_limit:
            self._clear_prefill_route_samples()
            self.state = EPLBManagerState.COLLECTING
        else:
            # rank 0 的路由分布足以代表全局分布，因此只使用 rank 0 的本地
            # 样本判断统计量是否充足，再广播布尔决策以保持所有 rank 的状态
            # 转移一致。样本不足时不启动大块原始负载 all-gather。
            has_enough_load = None
            if self.global_rank == 0:
                average_tokens_per_expert = local_load.sum().item() / local_load.numel()
                has_enough_load = average_tokens_per_expert >= EPLB_MIN_AVERAGE_TOKENS_PER_EXPERT
                if not has_enough_load:
                    logger.info(
                        "eplb continue collecting average_tokens_per_expert=%.2f threshold=%s",
                        average_tokens_per_expert,
                        EPLB_MIN_AVERAGE_TOKENS_PER_EXPERT,
                    )

            values = [has_enough_load]
            dist.broadcast_object_list(
                values,
                src=0,
                group=self.control_group,
            )
            has_enough_load = values[0]
            assert has_enough_load is not None

            if has_enough_load:
                self._load_gather_task = EPLBLoadGatherTask(
                    local_load_samples=local_load_samples,
                    load_gather_group=self.load_gather_group,
                )
                self._load_gather_task.start()
                self.state = EPLBManagerState.WAIT_LOAD_GATHER_FINISHED
            else:
                self.state = EPLBManagerState.COLLECTING

    def _step_wait_load_gather_finished(self) -> None:
        """等待原始负载汇集完成，并准备 planner 和 metrics 输入。"""
        assert self._load_gather_task is not None
        if not self._all_ranks_finished(self._load_gather_task.is_finished()):
            return

        gathered_load_samples = self._load_gather_task.result
        assert gathered_load_samples is not None
        assert gathered_load_samples.ndim == 4
        assert gathered_load_samples.shape[0] == self.world_size
        assert gathered_load_samples.shape[1] == len(self._eplb_impls)
        assert gathered_load_samples.shape[3] == self.num_logical_experts
        self._load_gather_task = None

        # gathered_load_samples 保留 [rank, layer, sample, logical_expert]
        # 原始结构并
        # 直接交给 planner。重排计算指标应表示一次真实 prefill 的
        # 关键路径开销，而不是多个不同批次累加后的虚拟大批次。
        # EP rank 以相同顺序执行 prefill，每次 dispatch 都将同一 sample
        # index 推进一次；因此各 rank 的第 0 行属于同一采样位置。
        # metrics 固定取该行，只汇总其在各 rank 上的分片，得到
        # [layer, logical_expert]。
        if self.global_rank == 0:
            self._pending_plan_load_samples = gathered_load_samples
            metric_sample_load_by_rank = gathered_load_samples[:, :, 0, :]
            self._metric_sample_load = metric_sample_load_by_rank.sum(dim=0)
        self.state = EPLBManagerState.PLAN_PLACEMENT

    def _step_plan_placement(self) -> None:
        """由 rank 0 使用已汇集的全局负载启动异步规划。"""
        self.state = EPLBManagerState.WAIT_PLAN_PLACEMENT_FINISHED
        if self.global_rank == 0:
            assert self._pending_plan_load_samples is not None
            # planner 消费保留 rank/sample 维的原始快照；指标使用其中
            # 一个 sample 行，确保 before/after 比较的是同一批负载。
            self._plan_task = EPLBPlanTask(
                planner=self.planner,
                logical_expert_load_samples=self._pending_plan_load_samples,
                current_placement=self.current_placement,
            )
            self._plan_task.start()
            # 任务对象已持有 Tensor，manager 不再保留重复引用。
            self._pending_plan_load_samples = None

    def _step_wait_plan_placement_finished(self) -> None:
        """等待 rank 0 完成规划并广播目标专家排布。"""
        placement: Optional[ExpertPlacement] = None
        if self.global_rank == 0:
            assert self._plan_task is not None
            if self._plan_task.is_finished():
                placement = self._plan_task.result
                assert placement is not None

        values = [placement]
        dist.broadcast_object_list(values, src=0, group=self.control_group)
        placement = values[0]
        if placement is None:
            return

        if self.global_rank == 0:
            assert self._metric_sample_load is not None
            eplb_metrics.publish_rebalance_compute_metrics(
                metric_client=self.metric_client,
                sample_load=self._metric_sample_load,
                current_placement=self.current_placement,
                target_placement=placement,
                expert_alignment=EPLB_EXPERT_ALIGNMENT,
            )
            self._metric_sample_load = None
            self._plan_task = None

        if placement == self.current_placement:
            if self.global_rank == 0:
                logger.info("eplb skip rearrangement because placement is unchanged")
            self.state = EPLBManagerState.COLLECTING
            return

        # 广播得到的 placement 已经是规划器新建的完整布局，没有外部持有者会
        # 再修改它，因此可直接保存，不需要逐层深拷贝。
        self.target_placement = placement
        self.state = EPLBManagerState.PLAN_TRANSFER

    def _step_plan_transfer(self) -> None:
        """启动异步传输规划，再进入完成状态轮询阶段。"""
        # 所有 rank 使用相同的 current/target placement 独立生成确定性的传输
        # 批次，避免广播体积较大的任务列表；耗时的逐层依赖分析放到后台线程，
        # 当前推理线程从下一次安全边界开始只需轮询完成状态。
        assert self.target_placement is not None
        self._transfer_planner = EPLBTransferPlanner(
            current_placement=self.current_placement,
            target_placement=self.target_placement,
            num_logical_experts=self.num_logical_experts,
            world_size=self.world_size,
            transfer_layer_parallelism=self.transfer_layer_parallelism,
        )
        self._transfer_planner.start()
        self.state = EPLBManagerState.WAIT_PLAN_TRANSFER_FINISHED

    def _step_wait_plan_transfer_finished(self) -> None:
        """等待所有 rank 异步生成相同的传输批次，再统一进入传输状态。"""
        assert self._transfer_planner is not None

        # 即使本 rank 已经完成，也必须等待其他 rank 的镜像任务列表就绪；否则
        # 提前进入 TRANSFERRING 的 rank 可能发起尚无对端参与的点对点传输。
        if not self._all_ranks_finished(self._transfer_planner.is_finished()):
            return

        pending_transfer_batches = self._transfer_planner.result
        assert pending_transfer_batches is not None
        self.pending_transfer_batches = pending_transfer_batches
        self._transfer_planner = None
        if not self.pending_transfer_batches:
            raise RuntimeError("planned EPLB rearrangement must contain at least one transfer")

        self.state = EPLBManagerState.TRANSFERRING
        if self.global_rank == 0:
            assert self.target_placement is not None
            changed_layer_count = sum(
                current != target for current, target in zip(self.current_placement, self.target_placement)
            )
            logger.info(
                "eplb started steps=%s changed_layer_count=%s changed_slot_count=%s",
                self.steps,
                changed_layer_count,
                sum(len(transfer_batch) for transfer_batch in self.pending_transfer_batches),
            )

    def _step_transferring(self) -> None:
        """启动或轮询一个传输批次；整批完成后再统一提交。"""
        if self.rebalance_started_at is None:
            self.rebalance_started_at = time.monotonic()

        # 没有活动批次时，所有 rank 根据相同的 pending 列表构造下一批任务。
        if self.active_transfer_batch is None:
            assert self.pending_transfer_batches is not None
            transfer_batch = self.pending_transfer_batches.pop(0) if self.pending_transfer_batches else []

            # 空批次表示公共任务列表已经耗尽，所有 rank 可以同时结束重排。
            if not transfer_batch:
                assert self.target_placement is not None
                self.current_placement = self.target_placement
                elapsed = time.monotonic() - self.rebalance_started_at
                if self.global_rank == 0:
                    self._persist_current_placement()
                self._clear_prefill_route_samples()
                self.completed_rebalance_count += 1
                self.pending_transfer_batches = None
                self.target_placement = None
                self.rebalance_started_at = None
                self.state = EPLBManagerState.COLLECTING
                if self.global_rank == 0:
                    logger.info(
                        "eplb completed wall_time=%.2fs completed_rebalance_count=%s max_rebalance_count=%s",
                        elapsed,
                        self.completed_rebalance_count,
                        self.max_rebalance_count,
                    )
            else:
                self.active_transfer_batch = transfer_batch

                # 普通批次允许多个 rank 不冲突的任务并行，但每个 rank 最多参与
                # 一条；覆盖环批次可能要求同一 rank 同时保存多个源/目标的 pinned
                # row，必须等整批传输完成后再统一覆盖 live 权重。
                self.active_transfers = [
                    PinnedMemoryEPLBTransfer(
                        weights=self._weights,
                        transfer_group=self.transfer_group,
                        current_global_rank=self.global_rank,
                        transfer_info=transfer_info,
                    )
                    for transfer_info in transfer_batch
                    if self.global_rank in (transfer_info.source_rank, transfer_info.dest_rank)
                ]
                for transfer in self.active_transfers:
                    transfer.start()
        else:
            # 已有活动批次时，本 step 只负责轮询；整批完成后才统一提交。
            self._poll_transfer_batch()

    def _poll_transfer_batch(self) -> None:
        """等待当前批次全部完成，随后统一提交并释放本地任务。"""
        # 每个 rank 只负责自己参与的任务；不参与当前批次的 rank，其本地任务
        # 列表为空，all([]) 自然为 True。所有 rank 汇总一个布尔值即可判断整批
        # 是否完成，无需重复传输并逐条匹配 EPLBTransferInfo。
        assert self.active_transfers is not None
        if not self._all_ranks_finished(all(transfer.is_finished() for transfer in self.active_transfers)):
            return

        # 所有 rank 使用相同的批次顺序提交，因此全局 placement 和 metadata
        # 始终一致；只有 destination rank 会额外写入实际专家权重。
        from lightllm.server.router.model_infer.infer_batch import g_infer_context

        # 专家权重和路由 metadata 都由 overlap stream 上的 MoE forward
        # 读取。将整批写操作排到同一条 stream，便可自然等待此前的 forward，
        # 并保证后续 forward 只能看到完整提交后的权重与 metadata。
        assert self.active_transfer_batch is not None
        with torch.cuda.stream(g_infer_context.get_overlap_stream()):
            for transfer_info in self.active_transfer_batch:
                self._commit_transfer(transfer_info)
            for layer_index in {transfer_info.layer_index for transfer_info in self.active_transfer_batch}:
                self._publish_layer_metadata(layer_index)

        self.active_transfers = None
        self.active_transfer_batch = None

    def _commit_transfer(self, transfer_info: EPLBTransferInfo) -> None:
        """把一条已完成传输提交到 live 权重和完整布局。"""
        is_destination_rank = transfer_info.dest_rank == self.global_rank
        if is_destination_rank:
            assert self.active_transfers is not None
            active_transfer = next(
                (transfer for transfer in self.active_transfers if transfer.transfer_info == transfer_info),
                None,
            )
            assert active_transfer is not None, "EPLB destination rank has no matching completed transfer"
            for tensor_buffer in active_transfer.tensor_buffers:
                tensor_buffer.live_tensor[transfer_info.dest_local_expert_index].copy_(
                    tensor_buffer.pinned_row,
                    non_blocking=True,
                )

        layer_index = transfer_info.layer_index
        layer_impl = self._eplb_impls[layer_index]
        self.current_placement[layer_index][transfer_info.dest_rank][
            transfer_info.dest_local_expert_index
        ] = transfer_info.source_logical_expert_id
        if is_destination_rank:
            layer_impl.local_logics_expert_ids_list[
                transfer_info.dest_local_expert_index
            ] = transfer_info.source_logical_expert_id

    def _all_ranks_finished(self, local_finished: bool) -> bool:
        """通过控制通信组判断所有 rank 的当前后台任务是否完成。"""
        finished_by_rank = [False] * self.world_size
        dist.all_gather_object(
            finished_by_rank,
            local_finished,
            group=self.control_group,
        )
        return all(finished_by_rank)

    def _publish_layer_metadata(self, layer_index: int) -> None:
        """在整批槽位更新完成后发布该层路由 metadata。"""
        layer_impl = self._eplb_impls[layer_index]
        logical_to_physical_map = torch.tensor(
            build_logical_to_physical_map(
                self.current_placement[layer_index],
                self.num_logical_experts,
                current_rank=self.global_rank,
                node_world_size=self.node_world_size,
            ),
            dtype=torch.int32,
            pin_memory=True,
        )
        layer_impl.logical_to_physical_map.copy_(logical_to_physical_map, non_blocking=True)

    def _clear_prefill_route_samples(self) -> None:
        """在 overlap stream 上清空所有层的 prefill 路由样本和设备端索引。"""
        from lightllm.server.router.model_infer.infer_batch import g_infer_context

        # prefill route sample 由 forward 中的 Triton kernel 在 overlap stream 上更新。
        # 将 zero_ 排到同一条 stream，可保证它位于此前 forward 之后、下一次
        # forward 之前，无需额外 synchronize，也不会与 atomic add 并发。
        with torch.cuda.stream(g_infer_context.get_overlap_stream()):
            for impl in self._eplb_impls:
                impl.prefill_route_counter.zero_()
                impl.prefill_route_sample_index.zero_()

    def _persist_current_placement(self) -> None:
        """由 rank 0 将当前完整布局写回启动时指定的输入/输出文件。"""
        if self.config_path is None:
            return
        save_placement_config(
            self.config_path,
            layer_indexes=self.layer_indexes,
            placement=self.current_placement,
            num_logical_experts=self.num_logical_experts,
            world_size=self.world_size,
            num_redundant_experts_per_rank=self.num_redundant_experts_per_rank,
        )


def _find_fused_moe_weights(model: TpPartBaseModel) -> List[FusedMoeWeight]:
    weights: List[FusedMoeWeight] = []
    for layer in model.trans_layers_weight:
        weight = getattr(layer, "experts", None)
        if isinstance(weight, FusedMoeWeight) and weight.enable_ep_moe:
            weights.append(weight)
    return weights
