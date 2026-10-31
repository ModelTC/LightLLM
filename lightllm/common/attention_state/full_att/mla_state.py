from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, List
from math import prod
from dataclasses import dataclass

import torch

from lightllm.utils.envs_utils import get_env_start_args

from ..utils import deepcopy_result, dtype_name
from ..types import AxisMapping, TransferRegion
from .base_state import FullAttentionState
from .packed_transfer import transfer_packed_pages
from .dp_transfer import copy_packed_tokens

if TYPE_CHECKING:
    from lightllm.common.attention_state.types import StateTransferLayout


@dataclass(frozen=True)
class MLAAttentionConfig:
    """普通 MLA 的独立 latent 布局；单份 latent 固定在各 TP rank 完整复制。"""

    # 外层顺序对应物理存储层，组内逻辑层共享该方案的全部字段；各组非空，逻辑层不能重复。
    # 例如 ((0, 4), (8,)) 分配两层，逻辑层 0/4 共用第一层的全部字段。
    # 分组不决定执行顺序或写入者；模型须让共享消费者跳过写入，避免覆盖生产层数据。
    logical_layer_groups: tuple[tuple[int, ...], ...]
    # 编码前的完整 latent 宽度，包含模型需要的 RoPE 部分，不含 indexer 或 padding。
    latent_dim: int
    # 未量化 latent 的浮点精度，也作为 MLA 工厂选择 packed 编码前的原始精度。
    dtype: torch.dtype
    # 所属模型 TP 组内的 rank；各 rank 保存相同 latent，仅 rank 0 写出共享外部页。
    tp_rank: int = 0
    # 保存完整 latent 副本的 TP rank 数，不改变 latent 的维度或物理行宽。
    tp_world_size: int = 1

    def __post_init__(self) -> None:
        assert isinstance(self.logical_layer_groups, tuple) and all(
            isinstance(group, tuple) and group for group in self.logical_layer_groups
        ), "layer groups must be a tuple of nonempty tuples"
        layers = [layer for group in self.logical_layer_groups for layer in group]
        assert all(isinstance(layer, int) and layer >= 0 for layer in layers), "logical layer IDs must be nonnegative integers"
        assert len(layers) == len(set(layers)), "a logical layer must belong to exactly one storage group"
        assert self.latent_dim > 0 and 0 <= self.tp_rank < self.tp_world_size


class MLAFullAttentionState(FullAttentionState):
    """存储复制到各 TP rank 的 MLA 压缩 latent，使用单个 KV 字段。"""

    # ========================================================================
    # 一、资源定义与初始化
    # ========================================================================

    def __init__(
        self,
        layout_config: MLAAttentionConfig,
        page_size: int = 1,
        device: str | torch.device = "cuda",
    ) -> None:
        """按 MLA 布局配置准备压缩 KV 存储。"""
        self.config = layout_config
        self.state_name: str | None = None  # 由 manager 绑定模型角色时设置。
        # 外层下标是物理层位置；组内多个逻辑层映射到同一份 KV/量化字段存储。
        self._logical_to_storage_layer = {
            logical_layer_id: storage_layer_index
            for storage_layer_index, logical_layer_group in enumerate(layout_config.logical_layer_groups)
            for logical_layer_id in logical_layer_group
        }
        self.page_size = page_size
        self.device = torch.device(device=device)
        self.capacity = 0
        self.storage: torch.Tensor | None = None

    def get_state_type(self) -> str:
        """返回 MLA 压缩 KV 精度对应的存储方案标记。"""
        return "mla_" + dtype_name(dtype=self.config.dtype)

    def allocate_storage(self, token_capacity: int) -> None:
        """按统一 token 容量分配本 State 的独立 KV 存储与 HOLD 页。"""
        self.capacity = token_capacity
        self.storage = torch.empty(
            size=(
                len(self.config.logical_layer_groups),
                token_capacity + self.page_size,
                1,
                self.config.latent_dim,
            ),
            dtype=self.config.dtype,
            device=self.device,
        )

    def get_token_bytes(self) -> int:
        """由配置计算当前 TP rank 的单 token latent 字节数，无需分配存储。"""
        config = self.config
        return len(config.logical_layer_groups) * config.latent_dim * config.dtype.itemsize

    # ========================================================================
    # 二、Full attention：计算视图、KV 写入与完整 token 复制
    # ========================================================================

    def contains_logical_layer(self, logical_layer_id: int) -> bool:
        """判断指定逻辑层是否属于当前 State 的层映射，共享层同样返回 True。"""
        return logical_layer_id in self._logical_to_storage_layer

    def get_storage_layer_index(self, logical_layer_id: int) -> int:
        """按逻辑层号查询存储及配套字段的物理行；共享逻辑层返回同一下标。"""
        return self._logical_to_storage_layer[logical_layer_id]

    def get_layer_view(self, logical_layer_id: int) -> torch.Tensor:
        """按逻辑层号返回压缩 latent 视图，不拆分为 GQA 的 K/V head。"""
        return self.storage[self.get_storage_layer_index(logical_layer_id=logical_layer_id)]

    def write_full_att_kv(self, logical_layer_id: int, token_indices: torch.Tensor, kv: torch.Tensor) -> None:
        """通过 CUDA 算子将形状匹配的 MLA latent 数据写入指定物理 token 位置。"""
        assert token_indices.is_cuda, "token indices must reside on CUDA"
        storage_layer_index = self.get_storage_layer_index(logical_layer_id=logical_layer_id)
        if not len(token_indices):
            return
        assert self.storage.is_cuda, "KV writes require CUDA storage"
        from lightllm.common.basemodel.triton_kernel.destindex_copy_kv import destindex_copy_kv

        destindex_copy_kv(K=kv, DestLoc=token_indices, Out=self.storage[storage_layer_index])

    def copy_full_att_tokens(self, src_indices: torch.Tensor, dst_indices: torch.Tensor) -> None:
        """复制完整 token 数据；索引使用 Tensor，源与目标位置由调用方保证不重叠。"""
        assert src_indices.is_cuda and dst_indices.is_cuda, "token indices must reside on CUDA"
        assert src_indices.ndim == dst_indices.ndim == 1 and src_indices.numel() == dst_indices.numel(), (
            "token copies require equal one-dimensional index tensors"
        )
        assert src_indices.dtype in (torch.int32, torch.int64) and dst_indices.dtype in (torch.int32, torch.int64), (
            "token indices must be integer tensors"
        )
        assert self.storage.is_cuda, "token copies require CUDA storage"
        if not src_indices.numel():
            return
        from lightllm.common.basemodel.triton_kernel.kv_move import copy_kv_buffer_to_kv_buffer

        # 直接使用调用方提供的 CUDA 索引，保留原 dtype 和 stride。
        # 按字节直接从源槽位写到目标槽位，覆盖量化 scale 和 packed padding，不创建中间 KV。
        # 调用方保证源、目标集合不相交，且目标索引不重复，避免并行读写竞争。
        copy_kv_buffer_to_kv_buffer(
            src_mem_index=src_indices,
            dst_mem_index=dst_indices,
            kv_buffer=self.storage,
        )

    # ========================================================================
    # 三、CPU Cache 数据交换
    # ========================================================================

    @deepcopy_result
    @lru_cache(maxsize=None)
    def get_cpu_transfer_regions(self) -> tuple[TransferRegion, ...]:
        """按全局 CPU 页大小返回字段区间与 TP 切片，作为 CPU 页容量计算依据。

        覆盖 KV、动态量化参数及 indexer 有效字段，不包含 GPU 行 padding 或静态标定参数。
        """
        page_tokens = get_env_start_args().cpu_cache_token_page_size
        return self._get_transfer_regions(page_tokens=page_tokens)

    def write_to_cpu(
        self,
        token_indices: torch.Tensor,
        cpu_page_buffers: torch.Tensor,
        cpu_page_indices: List[int],
        layout: StateTransferLayout,
        cpu_page_ready_flags: list[bool] | None = None,
    ) -> None:
        """按布局直接写出有效 token 及配套字段，不创建中间 KV 存储。"""
        assert cpu_page_buffers.device.type == "cpu", "CPU cache buffer must reside on CPU"
        transfer_packed_pages(
            kv_storage=self.storage,
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
        cpu_page_indices: List[int],
        token_indices: torch.Tensor,
        layout: StateTransferLayout,
    ) -> None:
        """按布局直接恢复有效 token 及配套字段，保留未参与的存储区域。"""
        assert cpu_page_buffers.device.type == "cpu", "CPU cache buffer must reside on CPU"
        transfer_packed_pages(
            kv_storage=self.storage,
            token_indices=token_indices,
            page_buffers=cpu_page_buffers,
            page_indices=cpu_page_indices,
            layout=layout,
            write=False,
        )

    # ========================================================================
    # 四、DP 迁移
    # ========================================================================

    def copy_tokens_from_dp(
        self,
        src_token_indices: torch.Tensor,
        dst_token_indices: torch.Tensor,
        remote_dp_full_att_state: MLAFullAttentionState,
    ) -> None:
        """复制其他 DP 同一 TP 分片的完整 token 数据，层顺序保持一致。"""
        assert type(remote_dp_full_att_state) is type(self) and remote_dp_full_att_state.config == self.config, (
            "DP copy requires the same encoding and equivalent TP state configurations"
        )
        assert src_token_indices.is_cuda and dst_token_indices.is_cuda, "DP token indices must reside on CUDA"
        assert src_token_indices.numel() == dst_token_indices.numel(), "source and destination token counts differ"
        copy_packed_tokens(
            src_storage=remote_dp_full_att_state.storage,
            dst_storage=self.storage,
            src_token_indices=src_token_indices,
            dst_token_indices=dst_token_indices,
        )

    # ========================================================================
    # 五、PD 分离传输
    # ========================================================================

    @deepcopy_result
    @lru_cache(maxsize=None)
    def get_pd_transfer_regions(self) -> tuple[TransferRegion, ...]:
        """按全局 PD 页大小返回字段与切片，交由管理器安排其在联合页中的位置。"""
        page_tokens = get_env_start_args().pd_kv_page_size
        return self._get_transfer_regions(page_tokens=page_tokens)

    def write_to_pd_page(
        self,
        token_indices: torch.Tensor,
        pd_page_buffers: torch.Tensor,
        pd_page_index: int,
        layout: StateTransferLayout,
    ) -> None:
        """按布局直接写出有效 token 及配套字段，不创建中间 KV 存储。"""
        assert pd_page_buffers.is_cuda, "PD pages must reside on CUDA"
        transfer_packed_pages(
            kv_storage=self.storage,
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
        """按布局直接恢复有效 token 及配套字段，保留未参与的存储区域。"""
        assert pd_page_buffers.is_cuda, "PD pages must reside on CUDA"
        transfer_packed_pages(
            kv_storage=self.storage,
            token_indices=token_indices,
            page_buffers=pd_page_buffers,
            page_indices=[pd_page_index],
            layout=layout,
            write=False,
        )

    # ========================================================================
    # 六、私有辅助实现
    # ========================================================================

    def _get_transfer_regions(self, page_tokens: int) -> tuple[TransferRegion, ...]:
        """各 rank 复制同一 latent，仅 rank 0 写出共享外部页，所有 rank 均可读回。"""
        config = self.config
        shape = (len(config.logical_layer_groups), page_tokens, 1, config.latent_dim)
        return (
            TransferRegion(
                field="kv",
                byte_start=0,
                byte_end=prod(shape) * config.dtype.itemsize,
                dtype=config.dtype,
                axes=("layer", "token", "head", "dim"),
                shape=shape,
                axis_mappings=(
                    AxisMapping(axis="head", local_start=0, external_start=0, length=1),
                    AxisMapping(axis="dim", local_start=0, external_start=0, length=config.latent_dim),
                ),
                write_enabled=config.tp_rank == 0,
            ),
        )
