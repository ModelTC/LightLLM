"""EPLB 专家传输计划的异步生成器。"""

from collections import deque
from typing import Iterator, List, Optional

from .async_task import EPLBAsyncTask
from .async_expert_transfer import EPLBTransferInfo, build_transfer_plan
from .placement import ExpertPlacement

TransferBatch = List[EPLBTransferInfo]
LayerTransferBatches = List[TransferBatch]


class EPLBTransferPlanner(EPLBAsyncTask):
    """在后台线程中生成传输计划，并把不同层的批次交错合并。

    输入布局在规划期间保持只读。每层仍独立调用 ``build_transfer_plan``，
    因此同一层内部的槽位覆盖依赖顺序不会改变；不同层之间没有权重和槽位
    依赖，可以把各层的下一批任务合并执行，以并行利用通信和拷贝资源。
    """

    def __init__(
        self,
        current_placement: ExpertPlacement,
        target_placement: ExpertPlacement,
        num_logical_experts: int,
        world_size: int,
        transfer_layer_parallelism: int,
    ) -> None:
        assert transfer_layer_parallelism > 0
        assert len(current_placement) == len(target_placement)
        self.current_placement = current_placement
        self.target_placement = target_placement
        self.num_logical_experts = num_logical_experts
        self.world_size = world_size
        self.transfer_layer_parallelism = transfer_layer_parallelism
        self.result: Optional[List[TransferBatch]] = None
        super().__init__(thread_name="eplb-transfer-plan")

    def execute(self) -> None:
        """逐层生成传输计划，再按配置的并行层数交错合并批次。"""
        # 三层列表分别表示 [layer][batch][transfer]。某层当前布局已经等于
        # 目标布局时，其 batch 列表为空；这里仍保留该层的位置，由后面的
        # 生成器统一识别并跳过空层。
        transfer_batches_by_layer: List[LayerTransferBatches] = []
        layer_placements = zip(self.current_placement, self.target_placement)
        for layer_index, (current_layer, target_layer) in enumerate(layer_placements):
            layer_transfer_batches = build_transfer_plan(
                current_placement=current_layer,
                target_placement=target_layer,
                layer_index=layer_index,
                num_logical_experts=self.num_logical_experts,
                world_size=self.world_size,
            )
            transfer_batches_by_layer.append(layer_transfer_batches)

        # 生成器每次 yield 的内容就是 manager 随后会并行启动、统一提交的一个
        # 全局传输批次。这里一次性转成列表，供所有 rank 按相同顺序执行。
        self.result = list(self._iter_parallel_transfer_batches(transfer_batches_by_layer))

    def _iter_parallel_transfer_batches(
        self,
        transfer_batches_by_layer: List[LayerTransferBatches],
    ) -> Iterator[TransferBatch]:
        """逐轮生成跨层合并批次，同时限制正在迁移的层数。

        每轮从前 ``transfer_layer_parallelism`` 个未完成层中各取一个小批次，
        合并后直接产出。仍有后续批次的层留到下一轮；已经完成的层被删除，
        后面的层自然向前补位。

        例如并行度为 2，四层批次数分别为 3、1、2、1，执行顺序为：

        ``(L0.0, L1.0) -> (L0.1, L2.0) -> (L0.2, L2.1) -> (L3.0)``
        """
        # 每层转换成独立队列，popleft() 表示取出该层“下一个允许执行”的小批次。
        # 这样不需要额外保存批次下标，也不会改变 build_transfer_plan 已经确定的
        # 层内覆盖顺序。无需迁移的空层在初始化时直接跳过。
        unfinished_layers = [deque(layer_batches) for layer_batches in transfer_batches_by_layer if layer_batches]
        while unfinished_layers:
            # 前 N 个未完成层构成本轮的并行层集合。每层只取一个小批次，因此
            # 同一层的后续依赖批次一定排在返回结果的下一轮或更晚位置。
            merged_batch: TransferBatch = []
            for layer_batches in unfinished_layers[: self.transfer_layer_parallelism]:
                merged_batch.extend(layer_batches.popleft())

            # 一个 yield 对应一次 manager 的“启动传输 -> 等待全部 rank 完成 ->
            # 统一提交”过程；不同层没有槽位依赖，可以安全合并到同一次执行。
            yield merged_batch

            # 清除已经没有后续批次的层。后面的未完成层会自动进入前 N 项，
            # 从而在下一轮补上空出的并行位置。
            unfinished_layers = [layer_batches for layer_batches in unfinished_layers if layer_batches]
