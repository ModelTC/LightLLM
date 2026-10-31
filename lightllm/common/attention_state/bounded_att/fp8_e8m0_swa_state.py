"""单个共享 KV head 的 FP8/E8M0 滑动窗口运行态，不依赖模型或 compressor。"""

from __future__ import annotations

from functools import lru_cache

from dataclasses import dataclass
from math import prod

import torch

from ..utils import deepcopy_result
from ..types import PageKind, StateTransferLayout, TensorField, TransferRegion
from .base_state import BoundedAttentionState
from .copy_kernel import copy_bounded_att_tensors
from .transfer_utils import bounded_att_buffer_copies
from .window_pages_pool import WindowPagesPool
from .fp8_e8m0_swa_transfer import copy_swa_window


@dataclass(frozen=True)
class FP8E8M0SlidingWindowConfig:
    # 本 State 服务的逻辑层，顺序对应物理 layer 轴，各层拥有独立窗口。
    logical_layer_ids: tuple[int, ...]
    # 每个 query 可见的近期 token 数，也是大小页和 PD 快照的逻辑窗口长度。
    window_size: int
    # 未编码 KV 向量维度，包含末尾保留 BF16 的 RoPE 部分。
    kv_dim: int
    # RoPE 维度；其余维度按 group64 FP8 编码，scale 使用 E8M0。
    rope_dim: int
    # packed 物理页内的 token 数；页内 KV 数据与 scale 分区存放。
    page_token_size: int = 128
    # 各 TP rank 完整复制 KV；仅 rank 0 写出共享 CPU/PD 区域。
    tp_rank: int = 0
    tp_world_size: int = 1

    @property
    def kv_data_bytes(self) -> int:
        return self.kv_dim + self.rope_dim

    @property
    def kv_bytes(self) -> int:
        scales = (self.kv_dim - self.rope_dim) // 64
        return self.kv_data_bytes + (scales + 7) // 8 * 8

    def get_packed_kv_bytes(self, token_count: int) -> int:
        """返回 packed 页字节数，包含数据行对齐。"""
        return (token_count * self.kv_bytes + self.kv_data_bytes - 1) // self.kv_data_bytes * self.kv_data_bytes

    def get_attention_view(self, packed_kv: torch.Tensor) -> torch.Tensor:
        """将运行态或 workspace 解释为 [page, token, head, byte]，不复制或重排数据。"""
        pages = packed_kv.view(-1, self.get_packed_kv_bytes(self.page_token_size))
        return pages[:, : self.page_token_size * self.kv_bytes].view(-1, self.page_token_size, 1, self.kv_bytes)

    def __post_init__(self) -> None:
        assert self.window_size > 0 and self.page_token_size > 0
        assert self.kv_dim > self.rope_dim >= 0 and (self.kv_dim - self.rope_dim) % 64 == 0
        assert self.logical_layer_ids and len(set(self.logical_layer_ids)) == len(self.logical_layer_ids)
        assert 0 <= self.tp_rank < self.tp_world_size
        assert self.get_packed_kv_bytes(self.page_token_size) % self.page_token_size == 0


class FP8E8M0SlidingWindowState(BoundedAttentionState):
    """仅保存近期原始 KV；prefill workspace 由计算 backend 管理。"""

    # ========================================================================
    # 一、资源定义与初始化
    # ========================================================================

    def __init__(
        self,
        config: FP8E8M0SlidingWindowConfig,
        request_capacity: int,
        mtp_step: int = 0,
        device: str | torch.device = "cuda",
    ) -> None:
        self.config = config
        self.state_name: str | None = None
        self.request_capacity = request_capacity
        self.mtp_step = mtp_step
        self._logical_to_storage_layer = {layer: index for index, layer in enumerate(config.logical_layer_ids)}
        page_tokens = config.page_token_size
        self.swa_capacity_tokens = (config.window_size + 2 * mtp_step + page_tokens - 1) // page_tokens * page_tokens
        self.HOLD_REQUEST_ID = request_capacity - 1
        self.HOLD_SWA_SLOT = self.HOLD_REQUEST_ID * self.swa_capacity_tokens
        self.swa_buffer = torch.zeros(
            size=(
                len(config.logical_layer_ids),
                request_capacity,
                self.swa_capacity_tokens,
                config.get_packed_kv_bytes(page_tokens) // page_tokens,
            ),
            dtype=torch.uint8,
            device=device,
        )
        self.big_pages_pool: WindowPagesPool | None = None
        self.small_pages_pool: WindowPagesPool | None = None

    def get_state_type(self) -> str:
        """窗口长度参与方案名称，允许同一模型组合不同滑窗。"""
        return f"swa_w{self.config.window_size}_fp8_group64_e8m0"

    def initialize_page_pools(
        self,
        big_page_capacity: int,
        small_page_capacity: int = 0,
        device: str | torch.device | int = "cpu",
        pin_memory: bool = True,
    ) -> None:
        """大小页只保存时间有序 KV，不保存 ring padding 或 MTP 候选位置。"""
        shape = (len(self.config.logical_layer_ids), self.config.window_size, self.config.kv_bytes)
        self.big_pages_pool = (
            WindowPagesPool(
                size=big_page_capacity,
                snapshot_shape=shape,
                dtype=torch.uint8,
                device=device,
                pin_memory=pin_memory,
            )
            if big_page_capacity
            else None
        )
        self.small_pages_pool = (
            WindowPagesPool(
                size=small_page_capacity,
                snapshot_shape=shape,
                dtype=torch.uint8,
                device=device,
                pin_memory=pin_memory,
            )
            if small_page_capacity
            else None
        )

    # ========================================================================
    # 二、运行态：层视图、重置与大小页保存恢复
    # ========================================================================

    def contains_logical_layer(self, logical_layer_id: int) -> bool:
        return logical_layer_id in self._logical_to_storage_layer

    def get_layer_view(self, logical_layer_id: int) -> torch.Tensor:
        """返回该层的 [request, ring_token, packed_byte] 视图。"""
        return self.swa_buffer[self._logical_to_storage_layer[logical_layer_id]]

    def reset_bounded_att_state(self, req_indices: list[int]) -> None:
        for req_idx in req_indices:
            self.swa_buffer[:, req_idx].zero_()

    def save_bounded_att_runtime_to_pages(
        self,
        req_indices: list[int],
        state_token_lens: list[int],
        page_kind: PageKind,
        page_slot_ids: list[int],
    ) -> None:
        pool = self._pool(page_kind=page_kind)
        for req_idx, length, slot in zip(req_indices, state_token_lens, page_slot_ids):
            self._copy_window(
                req_idx=req_idx,
                external=pool.buffer[slot],
                state_token_len=length,
                write=True,
            )

    def restore_bounded_att_runtime_from_pages(
        self,
        req_indices: list[int],
        state_token_lens: list[int],
        page_kind: PageKind,
        page_slot_ids: list[int],
    ) -> None:
        pool = self._pool(page_kind=page_kind)
        for req_idx, length, slot in zip(req_indices, state_token_lens, page_slot_ids):
            self._copy_window(
                req_idx=req_idx,
                external=pool.buffer[slot],
                state_token_len=length,
                write=False,
            )

    def commit_bounded_att_state(self, req_indices: torch.Tensor, accepted_steps: torch.Tensor) -> None:
        """ring 按绝对 token 位置寻址，无需提交额外游标。"""
        pass

    # ========================================================================
    # 三、CPU Cache：仅搬运时间有序的大小页快照
    # ========================================================================

    @deepcopy_result
    @lru_cache(maxsize=None)
    def get_cpu_transfer_regions(self) -> tuple[TransferRegion, ...]:
        """CPU 快照保存完整近期窗口。"""
        return self._get_transfer_regions()

    def write_state_pages_to_cpu(
        self,
        page_kind: PageKind,
        page_slot_ids: list[int],
        cpu_page_buffers: torch.Tensor,
        cpu_page_indices: list[int],
        layout: StateTransferLayout,
        cpu_page_ready_flags: list[bool] | None = None,
    ) -> None:
        """将指定快照写入 CPU 页，已就绪页保持不变。"""
        pool = self._pool(page_kind=page_kind)
        fields, pages = [], []
        for index, (slot, page) in enumerate(zip(page_slot_ids, cpu_page_indices)):
            if cpu_page_ready_flags is not None and cpu_page_ready_flags[index]:
                continue
            fields.append({"swa": TensorField(tensor=pool.buffer[slot], axes=("layer", "window", "channel"))})
            pages.append(page)
        for copies in bounded_att_buffer_copies(
            page_fields=fields,
            page_buffers=cpu_page_buffers,
            layout=layout,
            page_indices=pages,
            write=True,
        ):
            if copies:
                copy_bounded_att_tensors(copies=copies)

    def read_state_pages_from_cpu(
        self,
        cpu_page_buffers: torch.Tensor,
        cpu_page_indices: list[int],
        big_page_slot_ids: list[int],
        layout: StateTransferLayout,
    ) -> None:
        """CPU 页先恢复到大页池，随后再按前缀长度恢复请求 ring。"""
        fields = [
            {"swa": TensorField(tensor=self.big_pages_pool.buffer[slot], axes=("layer", "window", "channel"))}
            for slot in big_page_slot_ids
        ]
        for copies in bounded_att_buffer_copies(
            page_fields=fields,
            page_buffers=cpu_page_buffers,
            layout=layout,
            page_indices=cpu_page_indices,
            write=False,
        ):
            if copies:
                copy_bounded_att_tensors(copies=copies)

    # ========================================================================
    # 四、DP：相同方案的快照及请求运行态复制
    # ========================================================================

    def copy_big_pages_from_dp(
        self,
        src_page_ids: list[int],
        dst_page_ids: list[int],
        remote_dp_bounded_state: FP8E8M0SlidingWindowState,
    ) -> None:
        """复制已保存的大页，调用方保持槽位及 stream 依赖一致。"""
        assert self.config == remote_dp_bounded_state.config
        if self.big_pages_pool is not None:
            copy_bounded_att_tensors(
                copies=[
                    (remote_dp_bounded_state.big_pages_pool.buffer[src], self.big_pages_pool.buffer[dst])
                    for src, dst in zip(src_page_ids, dst_page_ids)
                ]
            )

    def copy_runtime_state_from_dp(
        self,
        src_req_indices: list[int],
        dst_req_indices: list[int],
        remote_dp_bounded_state: FP8E8M0SlidingWindowState,
    ) -> None:
        """直接复制相同配置和 MTP 容量的物理 ring，不创建快照。"""
        assert self.config == remote_dp_bounded_state.config and self.mtp_step == remote_dp_bounded_state.mtp_step
        copy_bounded_att_tensors(
            copies=[
                (remote_dp_bounded_state.swa_buffer[:, src], self.swa_buffer[:, dst])
                for src, dst in zip(src_req_indices, dst_req_indices)
            ]
        )

    # ========================================================================
    # 五、PD：按绝对前缀展开完整逻辑窗口，兼容不同 MTP 容量
    # ========================================================================

    @deepcopy_result
    @lru_cache(maxsize=None)
    def get_pd_transfer_regions(self) -> tuple[TransferRegion, ...]:
        """返回完整逻辑窗口，不包含本地 MTP 候选区。"""
        return self._get_transfer_regions()

    def write_to_pd_page(
        self,
        req_idx: int,
        pd_page_buffers: torch.Tensor,
        pd_page_index: int,
        state_token_len: int,
        layout: StateTransferLayout,
    ) -> None:
        """按绝对长度将请求 ring 写入选定 PD 页。"""
        page = pd_page_buffers[pd_page_index].view(torch.uint8).view(-1)
        for region in layout.regions:
            if region.write_enabled:
                external = page[region.byte_start : region.byte_end].view(region.dtype).view(region.shape)
                self._copy_window(
                    req_idx=req_idx,
                    external=external,
                    state_token_len=state_token_len,
                    write=True,
                )

    def read_from_pd_page(
        self,
        pd_page_buffers: torch.Tensor,
        req_idx: int,
        pd_page_index: int,
        state_token_len: int,
        layout: StateTransferLayout,
    ) -> None:
        """按接收端 ring 容量还原逻辑窗口，不写入大小页池。"""
        page = pd_page_buffers[pd_page_index].view(torch.uint8).view(-1)
        for region in layout.regions:
            external = page[region.byte_start : region.byte_end].view(region.dtype).view(region.shape)
            self._copy_window(
                req_idx=req_idx,
                external=external,
                state_token_len=state_token_len,
                write=False,
            )

    # ========================================================================
    # 六、内部页池和窗口搬运
    # ========================================================================

    def _pool(self, page_kind: PageKind) -> WindowPagesPool:
        assert page_kind in ("big", "small")
        pool = self.big_pages_pool if page_kind == "big" else self.small_pages_pool
        assert pool is not None, f"{page_kind} pages are disabled"
        return pool

    def _get_transfer_regions(self) -> tuple[TransferRegion, ...]:
        shape = (len(self.config.logical_layer_ids), self.config.window_size, self.config.kv_bytes)
        return (
            TransferRegion(
                field="swa",
                byte_start=0,
                byte_end=prod(shape),
                dtype=torch.uint8,
                axes=("layer", "window", "channel"),
                shape=shape,
                axis_mappings=(),
                write_enabled=self.config.tp_rank == 0,
            ),
        )

    def _copy_window(
        self,
        req_idx: int,
        external: torch.Tensor,
        state_token_len: int,
        write: bool,
    ) -> None:
        copy_swa_window(
            local=self.swa_buffer[:, req_idx],
            external=external,
            state_token_len=state_token_len,
            config=self.config,
            write=write,
        )
