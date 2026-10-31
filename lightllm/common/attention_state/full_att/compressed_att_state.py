"""序列压缩后的历史 KV，采用 FP8/E8M0、BF16 RoPE 与可选 indexer 的 packed 编码。"""

from __future__ import annotations

from functools import lru_cache

from typing import TYPE_CHECKING, List
from dataclasses import dataclass
from math import lcm, prod

import torch

from lightllm.utils.envs_utils import get_env_start_args

from ..utils import deepcopy_result
from ..types import AxisMapping, TransferRegion
from .base_state import FullAttentionState
from .packed_transfer import transfer_packed_pages
from .dp_transfer import copy_packed_tokens

if TYPE_CHECKING:
    from lightllm.common.attention_state.types import StateTransferLayout


@dataclass(frozen=True)
class CompressedAttentionConfig:
    # 外层顺序是物理 layer 轴；内层列出共享整份压缩 KV/indexer 存储的逻辑层。
    # 例如 ((0, 4), (8,)) 分配两份物理层存储，逻辑层 0/4 访问同一份数据。
    logical_layer_groups: tuple[tuple[int, ...], ...]
    # 以原始 token 为单位的压缩比例，不改变外部请求映射的索引域。
    compress_ratio: int
    # 每个压缩 KV 条目编码前的向量维度，包含末尾的 RoPE 部分；不是存储字节数。
    kv_dim: int
    # 保留 BF16 的 RoPE 维度；其余维度按每 64 个元素一组编码为 FP8，scale 使用 E8M0。
    rope_dim: int
    # 每个物理块覆盖的原始 token 数，必须能被 compress_ratio 整除。
    # 块内先保存全部 KV 数据，再保存 scale 和可选 indexer；不能按单个压缩条目拆开搬运。
    block_token_size: int
    # 可选 indexer key 的 FP8 维度，每个 key 另有一个 FP32 scale；0 表示不保存 indexer。
    indexer_dim: int = 0
    # 当前 State 所属 TP 组内的 rank；当前实现是单个共享 KV head，在各 rank 完整复制。
    # rank 0 写出共享 CPU/PD 页，每个 rank 均可读回，不按 TP 切分压缩条目的向量。
    tp_rank: int = 0
    # 当前模型 TP 组的 rank 数，只描述复制关系，不影响单个 rank 的存储字节数。
    tp_world_size: int = 1

    @property
    def kv_entry_bytes(self) -> int:
        """单条压缩 KV 的数据与 scale 字节数，不包含块末尾 padding 或 indexer。"""
        scale_count = (self.kv_dim - self.rope_dim) // 64
        return self.kv_dim + self.rope_dim + (scale_count + 7) // 8 * 8

    @property
    def kv_block_bytes(self) -> int:
        """块内 KV 数据及 E8M0 scale 区的字节跨度，包含数据行对齐，不包含 indexer。"""
        entries = self.block_token_size // self.compress_ratio
        # FP8 占一字节，BF16 RoPE 占两字节；每条目的 scale 区补齐到八字节。
        data_bytes = self.kv_dim + self.rope_dim
        return (entries * self.kv_entry_bytes + data_bytes - 1) // data_bytes * data_bytes

    @property
    def token_slot_bytes(self) -> int:
        """每个原始 token 索引摊销的物理字节数，包含 KV、scale、indexer 和块内 padding。"""
        block_bytes = self.kv_block_bytes
        if self.indexer_dim:
            entries = self.block_token_size // self.compress_ratio
            block_bytes += entries * (self.indexer_dim + torch.float32.itemsize)
        # 块既要能均摊到原始 token 索引域，也要保留 packed KV 数据行的对齐。
        alignment = lcm(self.block_token_size, self.kv_dim + self.rope_dim)
        return (block_bytes + alignment - 1) // alignment * alignment // self.block_token_size

    def get_attention_view(self, packed_kv: torch.Tensor) -> torch.Tensor:
        """返回算子使用的 [block, entry, head, byte] 视图，保留数据/scale 分区和块 stride。"""
        entries = self.block_token_size // self.compress_ratio
        return packed_kv[:, : entries * self.kv_entry_bytes].view(-1, entries, 1, self.kv_entry_bytes)

    def __post_init__(self) -> None:
        assert (
            self.compress_ratio > 0 and self.block_token_size > 0 and self.block_token_size % self.compress_ratio == 0
        )
        assert self.kv_dim > self.rope_dim >= 0 and (self.kv_dim - self.rope_dim) % 64 == 0
        assert self.indexer_dim >= 0 and 0 <= self.tp_rank < self.tp_world_size
        assert self.logical_layer_groups and all(self.logical_layer_groups), "storage layer groups must be nonempty"
        logical_layer_ids = [layer for group in self.logical_layer_groups for layer in group]
        assert len(logical_layer_ids) == len(
            set(logical_layer_ids)
        ), "logical layers cannot repeat across storage groups"


@dataclass(frozen=True)
class CompressedAttentionLayerView:
    # [压缩块, packed KV 字节]，保留整个块中数据与 scale 的分区格式。
    kv: torch.Tensor
    # 可选的 [压缩块, key/scale 字节]；未配置 indexer 时返回 None。
    indexer_k: torch.Tensor | None


class CompressedAttentionState(FullAttentionState):
    """保存一种序列压缩比例的历史 KV 和可选 indexer，不依赖具体模型配置。

    本 State 不执行压缩，不保存 SWA 或压缩器运行态；这些运行态由 bounded State 管理。
    不同压缩方案各自持有存储，沿用原始 token 索引及共享分配器。
    当前实现固定为单个共享 KV head，各 TP rank 完整复制；不执行 CSA/HCA 的历史选择。
    """

    # ========================================================================
    # 一、资源定义与初始化
    # ========================================================================

    def __init__(
        self,
        layout_config: CompressedAttentionConfig,
        page_size: int = 1,
        device: str | torch.device = "cuda",
    ) -> None:
        """只准备压缩 KV 配置和层映射，存储由统一容量分配阶段创建。"""
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
        """返回序列压缩比例、FP8 分组及 E8M0 scale 编码对应的方案标记。"""
        return f"compressed_c{self.config.compress_ratio}_fp8_group64_e8m0"

    def allocate_storage(self, token_capacity: int) -> None:
        """按统一 token 容量分配本 State 的独立 KV 存储与 HOLD 页。"""
        # 外部 allocator 的每个物理页必须包含完整压缩块，不能拆开块内的数据和 scale。
        assert (
            self.page_size > 0 and self.page_size % self.config.block_token_size == 0
        ), "allocator page_size must be a positive multiple of block_token_size"
        assert (
            token_capacity % self.config.block_token_size == 0
        ), "token capacity must contain complete compressed blocks"
        self.capacity = token_capacity
        # token 轴仍使用原始索引域；每 block_token_size 个槽位组成一个完整 packed 压缩块。
        self.storage = torch.zeros(
            size=(
                len(self.config.logical_layer_groups),
                token_capacity + self.page_size,
                1,
                self.config.token_slot_bytes,
            ),
            dtype=torch.uint8,
            device=self.device,
        )

    def get_token_bytes(self) -> int:
        """返回每个原始 token 索引的存储开销，包含全部物理层及压缩块 padding。"""
        config = self.config
        return len(config.logical_layer_groups) * config.token_slot_bytes

    # ========================================================================
    # 二、Full attention：计算视图、KV 写入与完整 token 复制
    # ========================================================================

    def contains_logical_layer(self, logical_layer_id: int) -> bool:
        """判断指定逻辑层是否属于当前 State 的层映射，共享层同样返回 True。"""
        return logical_layer_id in self._logical_to_storage_layer

    def get_storage_layer_index(self, logical_layer_id: int) -> int:
        """按逻辑层号查询存储及配套字段的物理行；共享逻辑层返回同一下标。"""
        return self._logical_to_storage_layer[logical_layer_id]

    def get_layer_view(self, logical_layer_id: int) -> CompressedAttentionLayerView:
        """从同一物理压缩块取 KV 和 indexer 视图，不重排数据或创建副本。"""
        config = self.config
        layer = self.storage[self.get_storage_layer_index(logical_layer_id=logical_layer_id)]
        blocks = layer.view(-1, config.block_token_size * config.token_slot_bytes)
        entries = config.block_token_size // config.compress_ratio
        kv_bytes = config.kv_block_bytes
        indexer = (
            blocks[:, kv_bytes : kv_bytes + entries * (config.indexer_dim + torch.float32.itemsize)]
            if config.indexer_dim
            else None
        )
        return CompressedAttentionLayerView(kv=blocks[:, :kv_bytes], indexer_k=indexer)

    def write_full_att_kv(
        self,
        logical_layer_id: int,
        token_indices: torch.Tensor,
        kv: torch.Tensor,
    ) -> None:
        """压缩器通过计算视图融合写入；普通 KV 写入路径不应调用此方法。"""
        raise NotImplementedError("compressed KV is written by the fused compressor")

    def copy_full_att_tokens(self, src_indices: torch.Tensor, dst_indices: torch.Tensor) -> None:
        """复制完整压缩块；调用方保证块内索引完整、顺序对应且源目标不重叠。"""
        assert src_indices.is_cuda and dst_indices.is_cuda, "token indices must reside on CUDA"
        assert (
            src_indices.ndim == dst_indices.ndim == 1 and src_indices.numel() == dst_indices.numel()
        ), "token copies require equal one-dimensional index tensors"
        # 数据、scale 和 indexer 在块内分区保存，不能只复制部分原始 token 槽位。
        assert (
            src_indices.numel() % self.config.block_token_size == 0
        ), "token copies require complete compressed blocks"
        assert src_indices.dtype in (torch.int32, torch.int64) and dst_indices.dtype in (
            torch.int32,
            torch.int64,
        ), "token indices must be integer tensors"
        assert self.storage.is_cuda, "token copies require CUDA storage"
        if not src_indices.numel():
            return
        copy_packed_tokens(
            src_storage=self.storage,
            dst_storage=self.storage,
            src_token_indices=src_indices,
            dst_token_indices=dst_indices,
        )

    # ========================================================================
    # 三、CPU Cache 数据交换
    # ========================================================================

    @deepcopy_result
    @lru_cache(maxsize=None)
    def get_cpu_transfer_regions(self) -> tuple[TransferRegion, ...]:
        """按全局 CPU 页大小描述完整压缩块，包含 KV、scale、indexer 和块内 padding。"""
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
        page_tokens = layout.tokens_per_page
        assert (
            page_tokens is not None and page_tokens > 0 and page_tokens % self.config.block_token_size == 0
        ), "CPU page token capacity must be a positive multiple of block_token_size"
        # 联合页可包含其他 State；只校验本 State 的区域完整且不越过单页末尾。
        page_bytes = prod(cpu_page_buffers.shape[1:]) * cpu_page_buffers.element_size()
        required_bytes = self.get_token_bytes() * page_tokens
        assert (
            0 <= layout.byte_start < layout.byte_start + required_bytes <= layout.byte_end <= page_bytes
        ), "compressed State byte range must fit within one CPU page"
        assert all(
            0 <= page < cpu_page_buffers.shape[0] for page in cpu_page_indices
        ), "CPU page index is out of bounds"
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
        page_tokens = layout.tokens_per_page
        assert (
            page_tokens is not None and page_tokens > 0 and page_tokens % self.config.block_token_size == 0
        ), "CPU page token capacity must be a positive multiple of block_token_size"
        # 读回也允许联合页中的非零偏移；页面必须完整容纳当前 State 的全部压缩块。
        page_bytes = prod(cpu_page_buffers.shape[1:]) * cpu_page_buffers.element_size()
        required_bytes = self.get_token_bytes() * page_tokens
        assert (
            0 <= layout.byte_start < layout.byte_start + required_bytes <= layout.byte_end <= page_bytes
        ), "compressed State byte range must fit within one CPU page"
        assert all(
            0 <= page < cpu_page_buffers.shape[0] for page in cpu_page_indices
        ), "CPU page index is out of bounds"
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
        remote_dp_full_att_state: CompressedAttentionState,
    ) -> None:
        """复制其他 DP 的完整压缩块；调用方保证块内索引完整且顺序对应。"""
        assert (
            type(remote_dp_full_att_state) is type(self) and remote_dp_full_att_state.config == self.config
        ), "DP copy requires the same encoding and equivalent TP state configurations"
        assert src_token_indices.is_cuda and dst_token_indices.is_cuda, "DP token indices must reside on CUDA"
        assert src_token_indices.numel() == dst_token_indices.numel(), "source and destination token counts differ"
        assert (
            src_token_indices.numel() % self.config.block_token_size == 0
        ), "DP token copies require complete compressed blocks"
        copy_packed_tokens(
            src_storage=remote_dp_full_att_state.storage,
            dst_storage=self.storage,
            src_token_indices=src_token_indices,
            dst_token_indices=dst_token_indices,
        )

    # ========================================================================
    # 五、PD 分离传输
    # ========================================================================

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
        # packed scale/indexer 可位于末尾有效 token 之后，PD 必须传完整的末压缩块。
        # TODO：将末块索引扩展融合到传输 kernel，省去这份很小的索引临时区。
        alignment = self.config.block_token_size
        padding = (-token_indices.numel()) % alignment
        if token_indices.numel() and padding:
            tail = token_indices[-1:] + torch.arange(
                start=1,
                end=padding + 1,
                dtype=token_indices.dtype,
                device=token_indices.device,
            )
            token_indices = torch.cat(tensors=(token_indices, tail))
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
        # packed scale/indexer 可位于末尾有效 token 之后，PD 必须传完整的末压缩块。
        # TODO：将末块索引扩展融合到传输 kernel，省去这份很小的索引临时区。
        alignment = self.config.block_token_size
        padding = (-token_indices.numel()) % alignment
        if token_indices.numel() and padding:
            tail = token_indices[-1:] + torch.arange(
                start=1,
                end=padding + 1,
                dtype=token_indices.dtype,
                device=token_indices.device,
            )
            token_indices = torch.cat(tensors=(token_indices, tail))
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
        """整个物理压缩块按原始 token 槽位传输，KV、scale、indexer 和 padding 一起保留。"""
        config = self.config
        assert (
            page_tokens > 0 and page_tokens % config.block_token_size == 0
        ), "CPU/PD page token capacity must be a positive multiple of block_token_size"
        shape = (len(config.logical_layer_groups), page_tokens, 1, config.token_slot_bytes)
        return (
            TransferRegion(
                field="kv",
                byte_start=0,
                byte_end=prod(shape),
                dtype=torch.uint8,
                axes=("layer", "token", "head", "dim"),
                shape=shape,
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
                        length=config.token_slot_bytes,
                    ),
                ),
                write_enabled=config.tp_rank == 0,
            ),
        )
