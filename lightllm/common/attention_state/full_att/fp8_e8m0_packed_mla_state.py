"""FP8/E8M0 packed MLA：448 维 FP8、64 维 BF16，并在同一行保存 indexer。"""

from __future__ import annotations

from functools import lru_cache

from dataclasses import dataclass
from typing import ClassVar

import torch

from lightllm.utils.envs_utils import get_env_start_args

from ..utils import deepcopy_result
from ..types import AxisMapping, StateTransferLayout, TransferRegion
from .base_state import FullAttentionState
from .dp_transfer import copy_packed_tokens
from .packed_transfer import transfer_packed_pages
from .packed_mla_state import PackedMLALayerView


@dataclass(frozen=True)
class FP8E8M0PackedMLAConfig:
    """固定 group64/E8M0 编码，只记录调用方能够选择的原子布局信息。"""

    # 外层为物理存储层，内层为共享该层全部 KV/indexer 的逻辑层号。
    logical_layer_groups: tuple[tuple[int, ...], ...]
    # 每个 token 的 indexer key 维度；其后另保存一个 FP32 scale，不包含在该维度中。
    indexer_dim: int = 128
    # 完整 latent 在各 TP rank 复制；rank 0 写出共享外部页，各 rank 均可读回。
    tp_rank: int = 0
    # 所属 TP 组大小，不增加物理 KV head，也不改变单 token 的存储字节数。
    tp_world_size: int = 1

    # 算子输入的完整 latent 维度，不是量化后的字节数，也不包含 indexer。
    LATENT_DIM: ClassVar[int] = 512
    # 固定编码：448 个 FP8 元素 + 64 个 BF16 元素 + 7 个 E8M0 scale + 1 字节对齐。
    # 末尾 64 维保留 BF16 不代表启用了 RoPE，GLM 使用 NoPE attention。
    KV_BYTES: ClassVar[int] = 584

    @property
    def payload_bytes(self) -> int:
        """一行有效数据的字节数：KV 编码、indexer key 及其 FP32 scale，不包含行尾 padding。"""
        return self.KV_BYTES + self.indexer_dim + torch.float32.itemsize

    @property
    def token_bytes(self) -> int:
        """每个 token 的物理行字节数，包含末尾 16 字节对齐 padding。"""
        return (self.payload_bytes + 15) // 16 * 16

    def __post_init__(self) -> None:
        assert self.logical_layer_groups and all(self.logical_layer_groups)
        layers = [layer for group in self.logical_layer_groups for layer in group]
        assert len(layers) == len(set(layers)), "logical layers cannot repeat across storage groups"
        assert self.indexer_dim > 0 and 0 <= self.tp_rank < self.tp_world_size


class FP8E8M0PackedMLAState(FullAttentionState):
    """独立的 group64/E8M0 MLA 实现，不继承其他具体 State 或动态选择编码。"""

    # ========================================================================
    # 一、资源定义与初始化
    # ========================================================================

    def __init__(
        self,
        layout_config: FP8E8M0PackedMLAConfig,
        page_size: int = 1,
        device: str | torch.device = "cuda",
    ) -> None:
        """只准备布局和逻辑层映射，实际存储由管理器确定 token 容量后统一分配。"""
        self.config = layout_config
        self.state_name: str | None = None
        self.page_size = page_size
        self.device = torch.device(device=device)
        self.capacity = 0
        self.storage: torch.Tensor | None = None
        self._logical_to_storage_layer = {
            logical_layer_id: storage_layer_index
            for storage_layer_index, group in enumerate(layout_config.logical_layer_groups)
            for logical_layer_id in group
        }

    def get_state_type(self) -> str:
        """返回固定算法、量化分组与 scale 编码标记，用于 main/draft 的规范成员名。"""
        return "mla_packed_fp8_group64_e8m0"

    def allocate_storage(self, token_capacity: int) -> None:
        """分配普通 token 及 HOLD 页；完整行包含 KV、indexer 和 16 字节对齐 padding。"""
        self.capacity = token_capacity
        self.storage = torch.empty(
            size=(len(self.config.logical_layer_groups), token_capacity + self.page_size, 1, self.config.token_bytes),
            dtype=torch.uint8,
            device=self.device,
        )

    def get_token_bytes(self) -> int:
        """返回当前 TP rank 的单 token 开销，包含全部物理层和行尾 padding。"""
        return len(self.config.logical_layer_groups) * self.config.token_bytes

    # ========================================================================
    # 二、Full attention：计算视图、KV 写入与完整 token 复制
    # ========================================================================

    def contains_logical_layer(self, logical_layer_id: int) -> bool:
        """判断该逻辑层是否使用当前 State，共享逻辑层同样返回 True。"""
        return logical_layer_id in self._logical_to_storage_layer

    def get_storage_layer_index(self, logical_layer_id: int) -> int:
        """将逻辑层映射为物理存储行；同组逻辑层返回相同下标。"""
        return self._logical_to_storage_layer[logical_layer_id]

    def get_layer_view(self, logical_layer_id: int) -> PackedMLALayerView:
        """返回固定 584 字节 KV 和 indexer key/scale 视图，保留物理行 stride。"""
        row = self.storage[self.get_storage_layer_index(logical_layer_id=logical_layer_id)]
        return PackedMLALayerView(
            kv=row[..., : self.config.KV_BYTES],
            indexer_k=row[..., self.config.KV_BYTES : self.config.payload_bytes],
        )

    def write_full_att_kv(
        self,
        logical_layer_id: int,
        token_indices: torch.Tensor,
        kv: torch.Tensor,
    ) -> None:
        """使用固定 group64/E8M0 算子量化并写入 KV，保留 indexer 和行尾 padding。"""
        from lightllm.common.attention_state.full_att.fp8_e8m0_packed_mla_write import (
            write_fp8_e8m0_packed_kv,
        )

        assert token_indices.is_cuda, "token indices must reside on CUDA"
        if token_indices.numel() == 0:
            return
        write_fp8_e8m0_packed_kv(
            kv=kv,
            token_indices=token_indices,
            packed_kv=self.get_layer_view(logical_layer_id=logical_layer_id).kv,
        )

    def copy_full_att_tokens(self, src_indices: torch.Tensor, dst_indices: torch.Tensor) -> None:
        """按 CUDA 索引复制完整物理行，包含 indexer、scale 和 padding；两组索引不得重叠。"""
        from lightllm.common.basemodel.triton_kernel.kv_move import copy_kv_buffer_to_kv_buffer

        assert src_indices.is_cuda and dst_indices.is_cuda, "token indices must reside on CUDA"
        if src_indices.numel() == 0:
            return
        copy_kv_buffer_to_kv_buffer(
            src_mem_index=src_indices,
            dst_mem_index=dst_indices,
            kv_buffer=self.storage,
        )

    # ========================================================================
    # 三、CPU Cache：按 token 搬运整份有效编码
    # ========================================================================

    @deepcopy_result
    @lru_cache(maxsize=None)
    def get_cpu_transfer_regions(self) -> tuple[TransferRegion, ...]:
        """返回 CPU 页中的完整有效 token 编码，包含 indexer，不包含 GPU 行尾 padding。"""
        return self._get_transfer_regions(page_tokens=get_env_start_args().cpu_cache_token_page_size)

    def write_to_cpu(
        self,
        token_indices: torch.Tensor,
        cpu_page_buffers: torch.Tensor,
        cpu_page_indices: list[int],
        layout: StateTransferLayout,
        cpu_page_ready_flags: list[bool] | None = None,
    ) -> None:
        """直接写出 KV、E8M0 scale 和 indexer；已就绪的 CPU 页面保持不变。"""
        assert cpu_page_buffers.device.type == "cpu", "CPU cache buffer must reside on CPU"
        transfer_packed_pages(
            kv_storage=self.storage[..., : self.config.payload_bytes],
            token_indices=token_indices,
            page_buffers=cpu_page_buffers,
            page_indices=cpu_page_indices,
            layout=layout,
            write=True,
            page_readies=cpu_page_ready_flags,
        )

    def read_from_cpu(
        self,
        cpu_page_buffers: torch.Tensor,
        cpu_page_indices: list[int],
        token_indices: torch.Tensor,
        layout: StateTransferLayout,
    ) -> None:
        """从 CPU 页恢复完整有效 token 编码，支持末页只恢复部分 token。"""
        assert cpu_page_buffers.device.type == "cpu", "CPU cache buffer must reside on CPU"
        transfer_packed_pages(
            kv_storage=self.storage[..., : self.config.payload_bytes],
            token_indices=token_indices,
            page_buffers=cpu_page_buffers,
            page_indices=cpu_page_indices,
            layout=layout,
            write=False,
        )

    # ========================================================================
    # 四、DP 迁移：复制相同编码的完整物理行
    # ========================================================================

    def copy_tokens_from_dp(
        self,
        src_token_indices: torch.Tensor,
        dst_token_indices: torch.Tensor,
        remote_dp_full_att_state: FP8E8M0PackedMLAState,
    ) -> None:
        """从其他 DP 的同一 TP 分片复制完整 token 数据，由调用方提供可访问的 CUDA 索引。"""
        assert (
            type(remote_dp_full_att_state) is type(self) and remote_dp_full_att_state.config == self.config
        ), "DP copy requires the same encoding and equivalent TP state configurations"
        assert src_token_indices.is_cuda and dst_token_indices.is_cuda, "DP token indices must reside on CUDA"
        copy_packed_tokens(
            src_storage=remote_dp_full_att_state.storage,
            dst_storage=self.storage,
            src_token_indices=src_token_indices,
            dst_token_indices=dst_token_indices,
        )

    # ========================================================================
    # 五、PD 分离：传输固定的完整 token 编码
    # ========================================================================

    @deepcopy_result
    @lru_cache(maxsize=None)
    def get_pd_transfer_regions(self) -> tuple[TransferRegion, ...]:
        """返回固定 FP8/E8M0 格式的 PD 页描述，由管理器安排联合页偏移。"""
        return self._get_transfer_regions(page_tokens=get_env_start_args().pd_kv_page_size)

    def write_to_pd_page(
        self,
        token_indices: torch.Tensor,
        pd_page_buffers: torch.Tensor,
        pd_page_index: int,
        layout: StateTransferLayout,
    ) -> None:
        """将完整有效 token 编码写入指定 PD 页，网络发送与同步由调用方负责。"""
        assert pd_page_buffers.is_cuda, "PD pages must reside on CUDA"
        transfer_packed_pages(
            kv_storage=self.storage[..., : self.config.payload_bytes],
            token_indices=token_indices,
            page_buffers=pd_page_buffers,
            page_indices=[pd_page_index],
            layout=layout,
            write=True,
        )

    def read_from_pd_page(
        self,
        pd_page_buffers: torch.Tensor,
        token_indices: torch.Tensor,
        pd_page_index: int,
        layout: StateTransferLayout,
    ) -> None:
        """从指定 PD 页恢复完整有效 token 编码，保留其他 token 与行尾 padding。"""
        assert pd_page_buffers.is_cuda, "PD pages must reside on CUDA"
        transfer_packed_pages(
            kv_storage=self.storage[..., : self.config.payload_bytes],
            token_indices=token_indices,
            page_buffers=pd_page_buffers,
            page_indices=[pd_page_index],
            layout=layout,
            write=False,
        )

    # ========================================================================
    # 六、私有辅助：固定 packed 编码作为一个区域传输
    # ========================================================================

    def _get_transfer_regions(self, page_tokens: int) -> tuple[TransferRegion, ...]:
        # 有效字节在每行内连续，KV、scale 和 indexer 无需分别发射 kernel。
        # GPU 行有 padding，通过实际 stride 跳过；外部页只保存 payload_bytes。
        return (
            TransferRegion(
                field="kv",
                byte_start=0,
                byte_end=len(self.config.logical_layer_groups) * page_tokens * self.config.payload_bytes,
                dtype=torch.uint8,
                axes=("layer", "token", "head", "dim"),
                shape=(len(self.config.logical_layer_groups), page_tokens, 1, self.config.payload_bytes),
                axis_mappings=(
                    AxisMapping(
                        axis="head",
                        local_start=0,
                        external_start=0,
                        length=1,
                    ),
                    AxisMapping(
                        axis="dim",
                        local_start=0,
                        external_start=0,
                        length=self.config.payload_bytes,
                    ),
                ),
                write_enabled=self.config.tp_rank == 0,
            ),
        )
