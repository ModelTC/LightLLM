from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import dataclass

import torch

from ..utils import dtype_name
from ..types import AxisSlice, TransferRegion
from .base_state import AttentionTailState
from .kpool_transfer import transfer_kpool_pages

if TYPE_CHECKING:
    from lightllm.common.att_resources.types import StateTransferLayout


@dataclass(frozen=True)
class KPoolStateConfig:
    # 本 State 管理的逻辑层号；当前每层独占一份 tail，列表顺序对应物理层轴，不要求编号连续。
    logical_layer_ids: tuple[int, ...]
    # K-pool 的池化步长/压缩比例，单位为原始 token；例如 4 表示每 4 个 token 形成一个池化组。
    # 无快照前缀恢复要求长度对齐该步长；ring 的实际槽位数为 pool_size + State.mtp_step。
    pool_size: int
    # 单个原始 indexer key 的特征维度，不是 head 数，也不包含 gate 部分。
    # ring 的 channel 轴拼接 key 和同维度的 gate，共 2 * head_dim 个元素。
    head_dim: int
    # 原始 key/gate ring 的存储精度，不代表池化后 packed indexer key 的量化编码类型。
    dtype: torch.dtype
    # 当前模型 TP 组内的 rank。K-pool tail 在组内完整复制，传输时仅 rank 0 写出。
    tp_rank: int = 0
    # 当前模型的 TP 组大小，用于约束 rank 范围；不按此数切分 tail 的 head_dim。
    # 外部传输布局只保留一份 pool_size 长的逻辑窗口，所有 rank 可恢复到自己的 ring。
    tp_world_size: int = 1

    def __post_init__(self) -> None:
        """构造配置时检查维度、层号和 TP 范围。"""
        assert min(self.pool_size, self.head_dim) > 0, "invalid K-pool dimensions"
        assert self.logical_layer_ids and len(set(self.logical_layer_ids)) == len(self.logical_layer_ids), (
            "tail layers must be nonempty and unique"
        )
        assert 0 <= self.tp_rank < self.tp_world_size, "invalid TP rank"


class KPoolAttentionTailState(AttentionTailState):
    """保存 K-pool 原始 key/gate ring，完整保留 pool_size + mtp_step 个槽位。"""

    # ========================================================================
    # 一、资源定义与初始化
    # ========================================================================

    def __init__(
        self,
        config: KPoolStateConfig,
        request_capacity: int,
        mtp_step: int = 0,
        device: str | torch.device = "cuda",
    ) -> None:
        """为各层、各请求创建原始 key/gate ring，包含 MTP 所需额外历史。"""
        assert request_capacity > 0 and mtp_step >= 0, "invalid request capacity or MTP step"
        self.config = config
        self.request_capacity = request_capacity  # 包含预留的 HOLD 请求槽位。
        self.mtp_step = mtp_step  # ring 额外保留的 MTP 历史槽位数。
        self.resource_id: str | None = None  # 由 manager 绑定模型角色时设置。
        self._logical_to_storage_layer = {
            logical_layer_id: storage_layer_index
            for storage_layer_index, logical_layer_id in enumerate(config.logical_layer_ids)
        }
        self.tail_buffer = torch.zeros(
            (
                len(config.logical_layer_ids),
                request_capacity,
                config.pool_size + mtp_step,
                2 * config.head_dim,
            ),
            dtype=config.dtype,
            device=device,
        )

    def get_state_type(self) -> str:
        """返回 K-pool 压缩比例及存储精度。"""
        return f"kpool_c{self.config.pool_size}_{dtype_name(dtype=self.config.dtype)}"

    # ========================================================================
    # 二、Attention tail：计算视图与无快照重置
    # ========================================================================

    def has_logical_layer(self, logical_layer_id: int) -> bool:
        """查询逻辑层归属；共享同一物理层的逻辑层均返回 True。"""
        return logical_layer_id in self._logical_to_storage_layer

    def get_layer_view(self, logical_layer_id: int) -> torch.Tensor:
        """返回该层所有请求的原生 ring 视图，末轴包含 key 和 gate。"""
        return self.tail_buffer[self._logical_to_storage_layer[logical_layer_id]]

    def reset_attention_tail_state(
        self,
        req_indices: list[int],
    ) -> None:
        """清零指定请求的完整 ring；调用方保证用于空前缀或对齐池化边界，不用于 MTP 回滚。"""
        assert isinstance(req_indices, list), "request indices must be a list"
        for req_idx in req_indices:
            self.tail_buffer[:, req_idx].fill_(0)
    # ========================================================================
    # 三、PD 分离传输
    # ========================================================================

    def get_pd_transfer_regions(self) -> tuple[TransferRegion, ...]:
        """描述最近 pool_size 个逻辑位置的 key/gate，由 TP rank 0 写出；不包含 MTP 额外槽位。"""
        config = self.config
        window_size = config.pool_size
        channels = 2 * config.head_dim
        tail_region = TransferRegion(
            field="tail",
            byte_start=0,
            byte_end=len(config.logical_layer_ids) * window_size * channels * config.dtype.itemsize,
            dtype=config.dtype,
            axes=("layer", "window", "channel"),
            shape=(len(config.logical_layer_ids), window_size, channels),
            axis_slices=(
                AxisSlice(
                    axis="window",
                    src_start=0,
                    dst_start=0,
                    length=window_size,
                    src_circular=True,
                ),
                AxisSlice(
                    axis="channel",
                    src_start=0,
                    dst_start=0,
                    length=channels,
                ),
            ),
            write_enabled=config.tp_rank == 0,
        )
        return (tail_region,)

    def write_to_pd_buffer(
        self,
        req_idx: int,
        pd_page_buffers: torch.Tensor,
        pd_page_index: int,
        state_token_len: int,
        layout: StateTransferLayout,
    ) -> None:
        """将 prefill 末尾的逻辑窗口写入 PD 页；state_token_len 为已执行的前缀长度。"""
        assert layout.resource_id == self.resource_id, "transfer layout does not describe this resource"
        transfer_kpool_pages(
            tail=self.tail_buffer[:, req_idx],
            page_buffers=pd_page_buffers,
            page_index=pd_page_index,
            state_token_len=state_token_len,
            layout=layout,
            write=True,
        )

    def read_from_pd_buffer(
        self,
        pd_page_buffers: torch.Tensor,
        req_idx: int,
        pd_page_index: int,
        state_token_len: int,
        layout: StateTransferLayout,
    ) -> None:
        """按前缀长度将逻辑窗口放回本地 ring，保留窗口外槽位，允许两端 MTP 步数不同。"""
        assert layout.resource_id == self.resource_id, "transfer layout does not describe this resource"
        transfer_kpool_pages(
            tail=self.tail_buffer[:, req_idx],
            page_buffers=pd_page_buffers,
            page_index=pd_page_index,
            state_token_len=state_token_len,
            layout=layout,
            write=False,
        )
