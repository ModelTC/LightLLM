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
from lightllm.common.eplb_utils import NamedTensor, extract_eplb_expert_tensors
from lightllm.utils.log_utils import init_logger

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

    ``layer_index`` 是专家权重在 EPLB 层列表中的下标。源 rank 使用
    ``source_logical_expert_id`` 定位当前本地物理行，目标 rank 从
    ``tensor_buffers`` 中读取传输完成的 pinned memory 数据。
    """

    source_rank: int
    layer_index: int
    source_logical_expert_id: int
    dest_rank: int


@dataclass(frozen=True)
class _ExpertSource:
    """一个逻辑专家当前可用的物理副本位置。"""

    rank: int
    local_expert_index: int


class TransferStatus(Enum):
    """异步传输线程的生命周期状态。"""

    IDLE = "idle"
    RUNNING = "running"
    SUCCEEDED = "succeeded"


def build_transfer_plan(
    current_placement: torch.Tensor,
    target_placement: torch.Tensor,
    layer_index: int,
    num_logical_experts: int,
    world_size: int,
    node_world_size: int,
) -> List[EPLBTransferInfo]:
    """根据新旧冗余专家分布生成确定性的传输计划。

    ``current_placement`` 和 ``target_placement`` 只描述冗余槽位，形状均为
    ``[world_size, num_redundant_slots]``。固定主专家不在这两个张量中，但始终
    可以作为数据源。返回结果只包含发生变化的目标槽位所需专家，每个
    :class:`EPLBTransferInfo` 只描述一个逻辑专家的传输。

    为同一个逻辑专家选择数据源时，依次考虑：

    1. 优先使用目标 rank 所在节点上的已有副本，避免跨节点传输；
    2. 均衡各个源 rank 承担的传输次数；
    3. 使用 rank 和本地行号做稳定排序，保证所有 rank 生成一致结果。
    """
    assert (
        tuple(current_placement.shape)
        == tuple(target_placement.shape)
        == (
            world_size,
            current_placement.shape[1],
        )
    )
    num_primary_experts_per_rank = num_logical_experts // world_size
    current_placement_by_rank: List[List[int]] = current_placement.tolist()
    target_placement_by_rank: List[List[int]] = target_placement.tolist()

    # 每个逻辑专家的主专家行永远存在，因此先把它加入候选源；当前仍存在的
    # 冗余副本也可以作为源，这样目标 rank 有机会直接使用同节点副本。
    source_candidates_by_expert: List[List[_ExpertSource]] = []
    for logical_expert_id in range(num_logical_experts):
        primary_rank, primary_local_expert_index = divmod(logical_expert_id, num_primary_experts_per_rank)
        source_candidates_by_expert.append([_ExpertSource(primary_rank, primary_local_expert_index)])
    for rank, redundant_expert_ids in enumerate(current_placement_by_rank):
        for redundant_slot_index, logical_expert_id in enumerate(redundant_expert_ids):
            source_candidates_by_expert[logical_expert_id].append(
                _ExpertSource(
                    rank=rank,
                    local_expert_index=num_primary_experts_per_rank + redundant_slot_index,
                )
            )

    num_transfers_by_source_rank: List[int] = [0] * world_size
    transfer_infos: List[EPLBTransferInfo] = []
    for destination_rank in range(world_size):
        for destination_slot_index, logical_expert_id in enumerate(target_placement_by_rank[destination_rank]):
            if logical_expert_id == current_placement_by_rank[destination_rank][destination_slot_index]:
                continue
            source = min(
                source_candidates_by_expert[logical_expert_id],
                key=lambda candidate: (
                    candidate.rank // node_world_size != destination_rank // node_world_size,
                    num_transfers_by_source_rank[candidate.rank],
                    candidate.rank,
                    candidate.local_expert_index,
                ),
            )
            num_transfers_by_source_rank[source.rank] += 1
            transfer_infos.append(
                EPLBTransferInfo(
                    source_rank=source.rank,
                    layer_index=layer_index,
                    source_logical_expert_id=logical_expert_id,
                    dest_rank=destination_rank,
                )
            )

    return transfer_infos


class PinnedMemoryEPLBTransfer:
    """在后台线程中传输一个逻辑专家的全部权重张量。

    只有源 rank 和目标 rank 参与 Gloo 点对点通信，具体的数据路径是：

    ``源 GPU 权重行 -> 源 rank 的 pinned CPU 行 -> 目标 rank 的 pinned CPU 行``

    如果源和目标是同一个 rank，则只执行 GPU 到 pinned CPU 的本地复制，不产生
    网络通信。其他 rank 不分配 pinned row，也不参与该专家的数据传输。

    本类只负责异步传输，不修改 live 权重，也不更新路由 metadata。传输成功后，
    ``status`` 会变为 :attr:`TransferStatus.SUCCEEDED`，收到的数据保存在
    ``tensor_buffers``。EPLBManager 在主循环的安全边界同步提交这些数据。

    每个对象只表示构造函数中 ``transfer_info`` 指定的一次传输。逻辑专家 ID
    在源 rank 上通过该层当前的本地专家列表解析为物理行；目标物理槽位不属于
    传输职责，由 manager 根据目标 placement 决定。
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
        self._local_logical_expert_ids: List[int] = layer_weight.fuse_moe_impl.local_logics_expert_ids_list
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
                source_local_expert_index: int = self._local_logical_expert_ids.index(
                    transfer_info.source_logical_expert_id
                )
                with torch.cuda.stream(self._device_to_host_stream):
                    for tensor_buffer in self.tensor_buffers:
                        tensor_buffer.pinned_row.copy_(
                            tensor_buffer.live_tensor[source_local_expert_index],
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
        源 rank、目标 rank、逻辑专家和张量名称，避免依赖张量列表的隐式顺序。
        """
        transfer_info: EPLBTransferInfo = self.transfer_info
        message_identity = (
            f"{transfer_info.layer_index}:"
            f"{transfer_info.source_rank}:"
            f"{transfer_info.dest_rank}:"
            f"{transfer_info.source_logical_expert_id}:"
            f"{tensor_name}"
        )
        return zlib.crc32(message_identity.encode("utf-8")) & 0x7FFFFFFF
