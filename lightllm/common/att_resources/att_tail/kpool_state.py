from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import dataclass

import torch

from ..utils import dtype_name
from ..types import AxisSlice, TransferRegion
from .base_state import AttentionTailState
from .transfer_utils import att_tail_buffer_copies

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
        *,
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
                AxisSlice(axis="window", src_start=0, dst_start=0, length=window_size),
                AxisSlice(axis="channel", src_start=0, dst_start=0, length=channels),
            ),
            write_enabled=config.tp_rank == 0,
        )
        return (tail_region,)

    def write_to_pd_buffer(
        self, req_idx: int, pd_page_buffers: torch.Tensor,
        *, pd_page_index: int, state_token_len: int, layout: StateTransferLayout,
    ) -> None:
        """将 prefill 末尾的逻辑窗口写入 PD 页；state_token_len 为已执行的前缀长度。"""
        copies = self._pd_buffer_copies(pd_page_buffers, pd_page_index, req_idx, state_token_len, layout, write=True)
        for src_tensor, dst_tensor in copies:
            dst_tensor.copy_(src_tensor, non_blocking=True)

    def read_from_pd_buffer(
        self, pd_page_buffers: torch.Tensor, req_idx: int,
        *, pd_page_index: int, state_token_len: int, layout: StateTransferLayout,
    ) -> None:
        """按前缀长度将逻辑窗口放回本地 ring，清零其余候选历史，允许两端 MTP 步数不同。"""
        # TODO：PD 恢复可按目标 ring 槽位一次写入有效历史或零，融合清零和逐窗口复制；
        # 避免同一 kernel 内先清零再恢复造成竞争，同时清除窗口外的旧候选历史。
        copies = self._pd_buffer_copies(pd_page_buffers, pd_page_index, req_idx, state_token_len, layout, write=False)
        self.tail_buffer[:, req_idx].fill_(0)
        for src_tensor, dst_tensor in copies:
            dst_tensor.copy_(src_tensor, non_blocking=True)

    # ========================================================================
    # 四、私有辅助实现
    # ========================================================================

    def _pd_buffer_copies(
        self,
        pd_page_buffers: torch.Tensor,
        pd_page_index: int,
        req_idx: int,
        state_token_len: int,
        layout: StateTransferLayout,
        *,
        write: bool,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """为 PD prefill 交接生成复制视图，允许 P/D 使用不同的 MTP 步数。

        为什么不能直接复制物理 ring：
        - 本地 ring 的容量为 P+S，其中 P=pool_size，S=本地 mtp_step；绝对位置 pos
          存在 pos % (P+S) 槽位。P/D 的 S 不同时，形状和同一位置对应的物理槽位都不同。
        - 即使 P 端没有执行 MTP decode，prefill 算子也按实际 ring 容量取模写入，
          最近 P 个位置不一定落在 ring 的前 P 个槽位，因此不能直接截取 [:P] 传输。
        - 外部页固定保存最近 P 个逻辑位置，按时间顺序排列，形状为 [layers, P, 2*head_dim]。
          这样 PD 布局不依赖两端的 S；D 端的额外候选历史由后续 MTP 执行产生。

        位置和复制方向：
        - state_token_len=L 是该 State 已执行完 prefill 的 token 数，不是最后一个 token
          的下标，也不包含刚采样但尚未执行的输出 token；来自 PD 任务的 end_kv_index。
        - 外部 window[i] 对应绝对位置 pos=L-P+i。write=True 从本地 ring 的
          pos % (P+S) 读取并写入 window[i]；write=False 按相同规则恢复到本地 ring。
        - 例如 P=4、L=11，窗口保存位置 [7,8,9,10]。S_p=0 时从槽位 [3,0,1,2]
          写出，S_d=2 时恢复到槽位 [1,2,3,4]；直接复制同名物理槽位会恢复错误历史。
        - L<P 时，窗口前部对应负位置：写出时补零，读回时忽略。L=0 时没有有效历史。
          TP 副本只由 rank 0 写出，读回时每个 rank 都恢复自己的完整窗口。

        调用前提与副作用：
        - 双方具有相同模型及 main/draft 资源集合，池化步长、层顺序和字段表示兼容；
          本接口只用于 prefill 终点交接，不用于迁移已经执行 MTP 后的候选或回退历史。
        - req_idx 是本进程的请求槽位；pd_page_index 选择页集合中的物理页。
          region 的字节区间已相对整页，不再额外叠加 layout.byte_start。
        - 返回有效位置的 (源 Tensor 视图, 目标 Tensor 视图)，由读写入口提交 copy_；
          短前缀写出时，底层 helper 会立即提交外部窗口无效位置的清零。
        - 本方法不清零目标 ring。read_from_pd_buffer 取得全部视图后先清零该请求的
          本地 ring，再执行有效位置复制，避免保留旧请求或旧 MTP 候选历史。
          所有 GPU 操作遵循当前 stream，调用方负责完成依赖及源、目标存储的生命周期。
        """
        assert layout.resource_id == self.resource_id, "transfer layout does not describe this resource"
        assert 0 <= layout.byte_start < layout.byte_end, "invalid State byte interval"
        assert state_token_len >= 0, "PD tail requires a nonnegative prefix length"
        # helper 按 L 和本地 ring 实际容量逐位置取视图，不创建索引 Tensor 或读取 GPU 游标。
        return att_tail_buffer_copies(
            self.tail_buffer[:, req_idx], pd_page_buffers, layout,
            page_index=pd_page_index, state_token_len=state_token_len, write=write,
        )
