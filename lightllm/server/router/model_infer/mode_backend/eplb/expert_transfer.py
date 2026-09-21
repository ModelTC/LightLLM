"""EPLB 专家权重的逐层迁移。"""

import os
import threading
import zlib
from dataclasses import dataclass
from enum import Enum
from typing import List, Sequence

import torch
import torch.distributed as dist

from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.fused_moe_weight import FusedMoeWeight
from lightllm.utils.log_utils import init_logger

from .eplb_utils import NamedTensor, extract_eplb_expert_tensors

logger = init_logger(__name__)


@dataclass(frozen=True)
class ExpertTensorBuffer:
    """一项 live 专家张量及其单专家 pinned memory 缓冲行。"""

    name: str
    live_tensor: torch.Tensor
    pinned_row: torch.Tensor


@dataclass(frozen=True)
class EPLBTransferInfo:
    """单个逻辑专家的一次传输描述。

    ``layer_index`` 和 ``source_logical_expert_id`` 标识需要传输的专家；
    ``source_rank``、``source_local_expert_index`` 描述当前物理槽，
    ``dest_rank``、``dest_local_expert_index`` 描述目标物理槽。
    """

    layer_index: int
    source_logical_expert_id: int
    source_rank: int
    source_local_expert_index: int
    dest_rank: int
    dest_local_expert_index: int


class TransferStatus(Enum):
    """异步传输线程的生命周期状态。"""

    IDLE = "idle"
    RUNNING = "running"
    SUCCEEDED = "succeeded"


class PinnedMemoryEPLBTransfer:
    """在后台线程中传输一个逻辑专家的全部权重张量。

    只有源 rank 和目标 rank 参与 Gloo 点对点通信，具体的数据路径是：

    ``源 GPU 权重行 -> 源 rank 的 pinned CPU 行 -> 目标 rank 的 pinned CPU 行``

    如果源和目标是同一个 rank，则只执行 GPU 到 pinned CPU 的本地复制，不产生
    网络通信。其他 rank 不分配 pinned row，也不参与该专家的数据传输。

    本类只负责异步传输，不修改 live 权重，也不更新路由 metadata。传输成功后，
    ``status`` 会变为 :attr:`TransferStatus.SUCCEEDED`，收到的数据保存在
    ``tensor_buffers``。EPLBManager 在主循环的安全边界同步提交这些数据。

    每个对象只表示构造函数中 ``transfer_info`` 指定的一次传输。源 rank 直接
    读取 ``source_local_expert_index`` 指定的物理行；目标物理槽位不属于传输
    职责，由 manager 根据目标 placement 决定。
    """

    def __init__(
        self,
        weights: Sequence[FusedMoeWeight],
        transfer_group: dist.ProcessGroup,
        current_global_rank: int,
        transfer_info: EPLBTransferInfo,
    ) -> None:
        self._p2p_group: dist.ProcessGroup = transfer_group
        self._is_source_rank: bool = current_global_rank == transfer_info.source_rank
        self._is_destination_rank: bool = current_global_rank == transfer_info.dest_rank
        self.transfer_info: EPLBTransferInfo = transfer_info

        layer_weight: FusedMoeWeight = weights[transfer_info.layer_index]
        # 提取该 MoE 层实际参与推理的专家张量，包括 w13/w2 的量化后权重
        # （或非量化权重），以及配套的 weight_scale、weight_zero_point 等量化
        # 信息。后续会为每项张量创建对应的 pinned row，确保专家状态完整迁移。
        named_live_tensors: List[NamedTensor] = extract_eplb_expert_tensors(layer_weight)
        self._device: torch.device = named_live_tensors[0][1].device

        # 只有源和目标 rank 需要保存该专家的 pinned row。源 rank 用它作为
        # send buffer，目标 rank 用它作为 recv buffer，并在传输完成后直接交给
        # manager 提交到 live 权重，避免无关 rank 分配同样大小的 pinned memory。
        self.tensor_buffers: List[ExpertTensorBuffer] = []
        if self._is_source_rank or self._is_destination_rank:
            for tensor_name, live_tensor in named_live_tensors:
                pinned_row = torch.empty(
                    tuple(live_tensor.shape[1:]),
                    dtype=live_tensor.dtype,
                    device="cpu",
                    pin_memory=True,
                )
                self.tensor_buffers.append(
                    ExpertTensorBuffer(
                        name=tensor_name,
                        live_tensor=live_tensor,
                        pinned_row=pinned_row,
                    )
                )
        self._device_to_host_stream: torch.cuda.Stream = torch.cuda.Stream(device=self._device)

        self.status: TransferStatus = TransferStatus.IDLE
        self._transfer_thread: threading.Thread = threading.Thread(
            target=self._run_transfer,
            name=f"eplb-transfer-layer-{transfer_info.layer_index}-expert-{transfer_info.source_logical_expert_id}",
            daemon=True,
        )

    def start(self) -> None:
        """启动构造函数中 transfer_info 描述的异步传输。"""
        assert self.status is TransferStatus.IDLE, "EPLB transfer has already been started"
        self.status = TransferStatus.RUNNING
        self._transfer_thread.start()

    def is_finished(self) -> bool:
        """返回后台传输是否已经成功完成。"""
        return self.status is TransferStatus.SUCCEEDED

    def _run_transfer(self) -> None:
        """把指定专家的全部权重行传输到各 rank 的 pinned memory。"""
        try:
            transfer_info: EPLBTransferInfo = self.transfer_info
            if self._is_source_rank:
                torch.cuda.set_device(self._device)
                with torch.cuda.stream(self._device_to_host_stream):
                    for tensor_buffer in self.tensor_buffers:
                        tensor_buffer.pinned_row.copy_(
                            tensor_buffer.live_tensor[transfer_info.source_local_expert_index],
                            non_blocking=True,
                        )
                # Gloo 读取 pinned row 前，源 rank 必须等待 GPU -> CPU 拷贝完成。
                self._device_to_host_stream.synchronize()

            if transfer_info.source_rank != transfer_info.dest_rank:
                if self._is_source_rank:
                    for tensor_buffer in self.tensor_buffers:
                        message_tag = self._build_p2p_message_tag(tensor_buffer.name)
                        dist.send(
                            tensor_buffer.pinned_row,
                            dst=transfer_info.dest_rank,
                            group=self._p2p_group,
                            tag=message_tag,
                        )
                elif self._is_destination_rank:
                    for tensor_buffer in self.tensor_buffers:
                        message_tag = self._build_p2p_message_tag(tensor_buffer.name)
                        dist.recv(
                            tensor_buffer.pinned_row,
                            src=transfer_info.source_rank,
                            group=self._p2p_group,
                            tag=message_tag,
                        )
            self.status = TransferStatus.SUCCEEDED
        except BaseException:
            logger.exception("EPLB transfer failed")
            os._exit(1)

    def _build_p2p_message_tag(self, tensor_name: str) -> int:
        """为当前专家张量生成 source 和 destination 一致的 Gloo 整数 tag。

        Python ``hash`` 会因进程随机种子不同而产生不同结果，因此这里使用稳定的
        CRC32，并限制到 Gloo 可安全使用的有符号 31 位整数范围。标识中包含层、
        逻辑专家、源物理槽、目标物理槽和张量名称，避免依赖张量列表的隐式顺序。
        """
        transfer_info: EPLBTransferInfo = self.transfer_info
        message_identity = (
            f"{transfer_info.layer_index}:"
            f"{transfer_info.source_logical_expert_id}:"
            f"{transfer_info.source_rank}:"
            f"{transfer_info.source_local_expert_index}:"
            f"{transfer_info.dest_rank}:"
            f"{transfer_info.dest_local_expert_index}:"
            f"{tensor_name}"
        )
        return zlib.crc32(message_identity.encode("utf-8")) & 0x7FFFFFFF


def build_transfer_plan(
    current_placement: Sequence[Sequence[int]],
    target_placement: Sequence[Sequence[int]],
    layer_index: int,
    num_logical_experts: int,
    world_size: int,
) -> List[List[EPLBTransferInfo]]:
    """生成一层中所有发生变化的专家传输批次。

    ``current_placement`` 和 ``target_placement`` 的形状均为
    ``[world_size, num_local_experts]``。所有物理槽位都允许变化，因此传输源
    必须从当前实际存在的副本中选择。

    规划分为三个阶段：

    1. 建立当前槽位索引，同时找出布局调整前后专家不变的稳定槽位；
    2. 为每个变化的目标槽位绑定一个确定的源槽位。优先使用稳定副本，因为
       这种源槽位永远不会被本轮迁移覆盖；没有稳定副本时，循环使用当前已有
       的各个副本，避免把全部读取集中到同一个 rank；
    3. 根据槽位覆盖依赖生成两类执行批次。目标槽位不再作为任何待处理任务源
       的任务属于“安全任务”，彼此不要求原子提交，可以继续拆成较小批次并
       提前 commit；若不存在安全任务，剩余依赖必然由一个或多个环组成，环内
       任务不可拆分，必须全部传入 pinned memory 后统一 commit。

    图中使用 ``[槽位:当前专家] --传输专家--> [目标槽位]`` 表示一条任务。

    链式依赖示例
    ------------
    当前布局和目标布局分别为：

    ``current: [A:e0] [B:e1] [C:e2] [D:e2]``
    ``target:  [A:e0] [B:e0] [C:e1] [D:e2]``

    需要执行的传输形成一条依赖链：

    ``[A:e0] --e0--> [B:e1] --e1--> [C:e2]``

    初始 ``source_slots={A, B}``，因此只有 C 可以覆盖。虽然 C 中的 e2 被
    覆盖，但 D 中仍有稳定的 e2；第一批执行 ``B --e1--> C`` 后，e1 已经在
    C 中建立新副本，B 才不再作为源。第二批再执行 ``A --e0--> B``，最终
    得到目标布局。

    这个过程同时保护传入和被覆盖的专家：如果 B 保存的是 e1 的最后一个在线
    副本，而目标布局仍要求保留 e1，那么一定存在一条以 B 为源的待处理任务。
    此时 ``B in source_slots``，任何以 B 为目标的任务都不会进入安全批次。只有
    e1 已经存在于其他稳定槽位，或者前一批已经为 e1 建立新位置后，B 才允许
    被覆盖。

    环形依赖示例
    ------------
    当前布局为 ``[A:e0] [B:e1] [C:e2]``，目标布局为
    ``[A:e1] [B:e2] [C:e0]``，依赖关系为：

    ``[A:e0] --e0--> [C:e2] --e2--> [B:e1] --e1--> [A:e0]``

    A、B、C 都既是源又是目标，不存在安全目标槽位。三条任务必须组成同一个
    批次：先将 e0、e1、e2 全部传入 pinned memory，等全部传输完成后再统一
    覆盖 A、B、C，最后发布新 metadata。

    返回值按执行顺序保存最终的 commit 批次：同一安全波次会按参与 rank 拆成
    若干小批次，每个 rank 在一个小批次中最多参与一条任务；不冲突的 rank 仍
    可并行传输，已完成的小批次也可以立即提交。环批次则始终包含完整环。这样
    普通批次在每个 rank 上最多缓存一个专家，只有环形依赖才需要同时缓存多个
    专家，同时仍保证任何源专家都不会在最后一次读取之前被覆盖。
    """
    assert world_size > 0
    assert num_logical_experts % world_size == 0
    assert len(current_placement) == len(target_placement) == world_size
    num_local_experts_per_rank = len(current_placement[0])
    assert all(len(row) == num_local_experts_per_rank for row in current_placement)
    assert all(len(row) == num_local_experts_per_rank for row in target_placement)
    assert all(0 <= expert < num_logical_experts for row in current_placement for expert in row)
    assert all(0 <= expert < num_logical_experts for row in target_placement for expert in row)

    # 阶段 1：记录每个逻辑专家当前位于哪些物理槽位，并单独记录不会变化的
    # 稳定槽位。槽位统一表示为 (rank, local_expert_index)。
    Slot = tuple[int, int]
    current_slots_by_expert: List[List[Slot]] = [[] for _ in range(num_logical_experts)]
    stable_slots_by_expert: List[List[Slot]] = [[] for _ in range(num_logical_experts)]
    for rank, (current_row, target_row) in enumerate(zip(current_placement, target_placement)):
        for local_expert_index, (current_expert, target_expert) in enumerate(zip(current_row, target_row)):
            slot = (rank, local_expert_index)
            current_slots_by_expert[current_expert].append(slot)
            if current_expert == target_expert:
                stable_slots_by_expert[current_expert].append(slot)
    assert all(current_slots_by_expert), "current placement must contain every logical expert"
    assert set(expert for row in target_placement for expert in row) == set(range(num_logical_experts))

    # 阶段 2：为每个变化的目标槽位绑定一个确定的当前源槽位。
    #
    # 稳定副本不会出现在任何任务的目标位置，因此可以反复读取而没有覆盖
    # 风险。只有不存在稳定副本时，才循环使用该专家当前已有的所有副本。
    # 源槽位完整保存在 transfer_info 中，后续依赖分析和实际传输共用同一份信息。
    source_use_count = [0] * num_logical_experts
    pending_transfers: List[EPLBTransferInfo] = []
    for destination_rank, (current_row, target_row) in enumerate(zip(current_placement, target_placement)):
        for destination_local_expert_index in range(num_local_experts_per_rank):
            current_expert_id = current_row[destination_local_expert_index]
            target_expert_id = target_row[destination_local_expert_index]
            if target_expert_id == current_expert_id:
                continue
            source_slots = stable_slots_by_expert[target_expert_id] or current_slots_by_expert[target_expert_id]
            source_slot = source_slots[source_use_count[target_expert_id] % len(source_slots)]
            source_use_count[target_expert_id] += 1
            transfer_info = EPLBTransferInfo(
                layer_index=layer_index,
                source_logical_expert_id=target_expert_id,
                source_rank=source_slot[0],
                source_local_expert_index=source_slot[1],
                dest_rank=destination_rank,
                dest_local_expert_index=destination_local_expert_index,
            )
            pending_transfers.append(transfer_info)

    # 阶段 3：按照槽位覆盖依赖，将任务拆成可安全提交的执行批次。
    transfer_batches: List[List[EPLBTransferInfo]] = []
    while pending_transfers:
        source_slots = {
            (transfer_info.source_rank, transfer_info.source_local_expert_index) for transfer_info in pending_transfers
        }

        # 3.1 收集当前拓扑层次的全部安全任务。source_slots 是当前仍需保护的
        # 槽位集合：只要某个槽位中的专家尚未完成最后一次读取，该槽位就仍在
        # 集合中，任何以它为目标的任务都不能提交。这同时保护了目标槽位里即将
        # 被覆盖的旧专家，避免其最后一个在线副本被提前删除。
        #
        # 目标槽位不在 source_slots 的任务可以并行传输，并在整批完成后统一
        # 提交。若旧专家仍需迁往其他位置，该目标槽位必然也是相应任务的源，
        # 因而不会在本轮被选中；若它不是源，则旧专家已经有其他可用副本。
        #
        # 这里不能在找到第一个任务后立即修改 source_slots。只有整批提交并从
        # pending 中移除后，下一层目标槽位才真正变得安全。
        safe_transfer_batch: List[EPLBTransferInfo] = []
        remaining_transfers: List[EPLBTransferInfo] = []
        for transfer_info in pending_transfers:
            destination_slot = (transfer_info.dest_rank, transfer_info.dest_local_expert_index)
            if destination_slot not in source_slots:
                safe_transfer_batch.append(transfer_info)
            else:
                remaining_transfers.append(transfer_info)

        if safe_transfer_batch:
            # 安全任务之间没有原子提交要求，但若同一 rank 在一个批次中参与
            # 多条任务，就会同时创建多份专家 pinned buffer。这里按 rank 冲突
            # 继续拆分：每个 rank 在一个小批次中最多参与一条任务，不冲突的
            # rank 仍可并行传输，从而兼顾吞吐和 pinned memory 峰值。
            unbatched_transfers = safe_transfer_batch
            while unbatched_transfers:
                current_batch: List[EPLBTransferInfo] = []
                occupied_ranks: set[int] = set()
                deferred_transfers: List[EPLBTransferInfo] = []

                # 顺序扫描尚未分组的任务：rank 不冲突的任务进入当前批次，
                # 冲突任务留到下一轮。每轮至少取出一个任务，因此一定结束。
                for transfer_info in unbatched_transfers:
                    participant_ranks = {transfer_info.source_rank, transfer_info.dest_rank}
                    if participant_ranks & occupied_ranks:
                        deferred_transfers.append(transfer_info)
                    else:
                        current_batch.append(transfer_info)
                        occupied_ranks.update(participant_ranks)

                transfer_batches.append(current_batch)
                unbatched_transfers = deferred_transfers

            pending_transfers = remaining_transfers
        else:
            # 3.2 没有叶子时，每个目标槽位也一定是某条任务的源槽位。每个目标
            # 槽位只有一条写入任务，因此此时源槽位也不会重复，剩余依赖图必然
            # 分解为若干互不相交的简单环。任选第一条任务的源槽位，沿着
            # source_slot -> destination_slot 追踪，回到起点便得到一个完整环。
            #
            # 这里有 N 个互不重复的目标槽位，并且没有安全任务意味着这 N 个
            # 目标都包含在 source_slots 中。source_slots 最多也只有 N 项，因此
            # 它必然恰好有 N 项，即每个源槽位只对应一个目标；先显式校验这个
            # 条件，再构造字典，不会因重复 key 丢失任务。
            #
            # 例如 ``S -> A、S -> B、A -> S`` 中，源集合只有 ``{S, A}``，
            # B 不在源集合中，所以 ``S -> B`` 会先作为安全任务移除；剩余的
            # ``S -> A、A -> S`` 才会进入这里，并且每个源都只对应一个目标。
            assert len(source_slots) == len(pending_transfers)
            transfer_by_source_slot = {
                (transfer_info.source_rank, transfer_info.source_local_expert_index): transfer_info
                for transfer_info in pending_transfers
            }

            first_transfer = pending_transfers[0]
            cycle_start_slot = (first_transfer.source_rank, first_transfer.source_local_expert_index)
            source_slot = cycle_start_slot
            cycle_batch: List[EPLBTransferInfo] = []
            cycle_source_slots: set[Slot] = set()

            # 从任意源槽位出发，当前任务的目标槽位就是下一条任务的源槽位；
            # 目标重新回到起点时，一个完整环便已经收集完成。
            while True:
                assert source_slot not in cycle_source_slots
                cycle_source_slots.add(source_slot)
                transfer_info = transfer_by_source_slot[source_slot]
                cycle_batch.append(transfer_info)
                destination_slot = (transfer_info.dest_rank, transfer_info.dest_local_expert_index)
                if destination_slot == cycle_start_slot:
                    break
                source_slot = destination_slot

            transfer_batches.append(cycle_batch)
            pending_transfers = [
                transfer_info
                for transfer_info in pending_transfers
                if (transfer_info.source_rank, transfer_info.source_local_expert_index) not in cycle_source_slots
            ]

    return transfer_batches
