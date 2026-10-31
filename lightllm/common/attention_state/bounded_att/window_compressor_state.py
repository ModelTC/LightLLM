"""固定窗口、固定产出步长的 attention 压缩器候选/gate 运行态。"""

from __future__ import annotations

from functools import lru_cache

from dataclasses import dataclass
from math import prod
from typing import Literal

import torch

from lightllm.utils.envs_utils import get_env_start_args

from ..utils import deepcopy_result
from ..types import PageKind, StateTransferLayout, TensorField, TransferRegion
from .base_state import BoundedAttentionState
from .copy_kernel import copy_bounded_att_tensors
from .transfer_utils import bounded_att_buffer_copies
from .window_pages_pool import WindowPagesPool
from .window_compressor_transfer import copy_compressor_window


@dataclass(frozen=True)
class WindowCompressorStateConfig:
    """按固定窗口和产出步长保存候选向量及 gate，供模型的压缩或池化算子使用。

    这里只描述运行态的存储和历史保留方式，具体聚合、归一化及量化由模型算子完成。
    window_size / compress_stride 分别表示窗口覆盖的原始 token 数和相邻产出的间隔：
    - GLM K-pool：W4/S4，每四个 key/gate 形成一组，无跨组重叠历史。
    - V4 C4：W8/S4，每四步产出一次，下一组仍需保留四个 token 的历史。
    - V4 C128：W128/S128，每 128 步产出一次，无跨组重叠历史。
    MTP 的额外候选槽位由 State.mtp_step 扩展，不计入窗口或产出步长。
    """

    # 本实例服务的逻辑层，顺序对应运行态 Tensor 的 layer 轴。
    # attention KV 候选与索引 key 候选分别创建实例，可使用不同窗口、维度和精度。
    logical_layer_ids: tuple[int, ...]
    # 单次聚合覆盖的原始 token 窗口，不包含 MTP 额外槽位。
    # 例如 K-pool 为 4，V4 C4 为 8；压缩比例同为 4 不代表窗口长度相同。
    window_size: int
    # 相邻产出之间经过的原始 token 数；K-pool / V4 C4 为 4，V4 C128 为 128。
    # 等于 window_size 时无重叠，小于时保留 window_size - compress_stride 个历史位置。
    # 大小页快照在该步长的边界保存；PD 可在任意前缀终点传输完整逻辑窗口。
    compress_stride: int
    # 每个原始 token 保存的候选向量宽度；gate 具有相同形状和存储精度。
    # K-pool 中等于 indexer key 的 head_dim；V4 W8/S4 为前后半窗口分别保存候选，
    # 使用 2 * 输出向量维度。因此它不一定等于最终压缩条目的维度，也不包含 gate 的宽度。
    candidate_dim: int
    # 候选数据的用途，仅用于区分规范 State 成员名，不选择聚合算法或改变存储和恢复行为。
    # kv 表示 attention KV 候选；indexer 表示索引 key 候选及其 gate，不代表整个 indexer。
    target: Literal["kv", "indexer"] = "kv"
    # 候选与 gate 使用相同精度，沿最后一轴拼接为 candidate|gate，共用一个 Tensor。
    # 此精度描述压缩器输入历史，不是产出 KV / indexer key 的量化编码；
    # GLM K-pool 随模型精度配置（通常 BF16），V4 的 KV/indexer 压缩器当前使用 FP32。
    # 当前接口不支持候选与 gate 使用不同 dtype。
    dtype: torch.dtype = torch.float32
    # 所属模型 TP 组内的 rank；运行态完整复制，仅 rank 0 向共享外部页写出。
    tp_rank: int = 0
    # 保存相同运行态副本的 TP rank 数，不用于切分 candidate_dim 或窗口长度。
    tp_world_size: int = 1

    @property
    def overlap_tokens(self) -> int:
        """组完成时仍需跨步保留的重叠历史：W8/S4 为 4，W128/S128 为 0。"""
        return self.window_size - self.compress_stride

    def __post_init__(self) -> None:
        assert self.window_size >= self.compress_stride > 0 and self.candidate_dim > 0
        assert self.logical_layer_ids and len(set(self.logical_layer_ids)) == len(self.logical_layer_ids)
        assert self.target in ("kv", "indexer") and self.dtype in (torch.float32, torch.float16, torch.bfloat16)
        assert 0 <= self.tp_rank < self.tp_world_size


class WindowCompressorState(BoundedAttentionState):
    """只保存窗口候选/gate，不保存压缩后的 KV 或 SWA。

    大小页在产出步长的边界保存重叠历史，PD 则保存完整窗口以支持任意前缀长度。
    无重叠的窗口在组边界不需要快照，因此不创建大小页池，也不占 CPU 联合页字节。
    """

    # ========================================================================
    # 一、资源定义与初始化
    # ========================================================================

    def __init__(
        self,
        config: WindowCompressorStateConfig,
        request_capacity: int,
        mtp_step: int = 0,
        device: str | torch.device = "cuda",
    ) -> None:
        self.config = config
        self.state_name: str | None = None
        self.request_capacity = request_capacity
        self.mtp_step = mtp_step
        self._logical_to_storage_layer = {layer: index for index, layer in enumerate(config.logical_layer_ids)}
        self.buffer = torch.zeros(
            size=(
                len(config.logical_layer_ids),
                request_capacity,
                config.window_size + mtp_step,
                2 * config.candidate_dim,
            ),
            dtype=config.dtype,
            device=device,
        )
        # HOLD 不得成为有效候选；正常请求在 reset 时初始化候选及 gate。
        self.buffer[:, request_capacity - 1, :, config.candidate_dim :].fill_(float("-inf"))
        self.big_pages_pool: WindowPagesPool | None = None
        self.small_pages_pool: WindowPagesPool | None = None

    def get_state_type(self) -> str:
        """用途、窗口、步长和精度共同标识可复用的压缩器方案。"""
        precision = {torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16"}[self.config.dtype]
        return (
            f"compressor_{self.config.target}_w{self.config.window_size}"
            f"_s{self.config.compress_stride}_{precision}"
        )

    def initialize_page_pools(
        self,
        big_page_capacity: int,
        small_page_capacity: int = 0,
        device: str | torch.device | int = "cpu",
        pin_memory: bool = True,
    ) -> None:
        """只为跨步重叠历史创建快照池；无重叠时保持 None。"""
        if not self.config.overlap_tokens:
            return
        shape = (len(self.config.logical_layer_ids), self.config.overlap_tokens, 2 * self.config.candidate_dim)
        self.big_pages_pool = (
            WindowPagesPool(
                size=big_page_capacity,
                snapshot_shape=shape,
                dtype=self.config.dtype,
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
                dtype=self.config.dtype,
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
        """返回该层 [request, window+MTP, candidate|gate] 的计算视图。"""
        return self.buffer[self._logical_to_storage_layer[logical_layer_id]]

    def reset_bounded_att_state(self, req_indices: list[int]) -> None:
        for req_idx in req_indices:
            self.buffer[:, req_idx].zero_()
            self.buffer[:, req_idx, :, self.config.candidate_dim :].fill_(float("-inf"))

    def save_bounded_att_runtime_to_pages(
        self,
        req_indices: list[int],
        state_token_lens: list[int],
        page_kind: PageKind,
        page_slot_ids: list[int],
    ) -> None:
        """边界必须完成一个产出步长；仅保留下一步仍需读取的窗口重叠部分。"""
        assert all(
            length % self.config.compress_stride == 0 for length in state_token_lens
        ), "prefix must close a compressor stride"
        if not self.config.overlap_tokens:
            return
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
        assert all(
            length % self.config.compress_stride == 0 for length in state_token_lens
        ), "prefix must close a compressor stride"
        if not self.config.overlap_tokens:
            return
        pool = self._pool(page_kind=page_kind)
        for req_idx, length, slot in zip(req_indices, state_token_lens, page_slot_ids):
            self._copy_window(
                req_idx=req_idx,
                external=pool.buffer[slot],
                state_token_len=length,
                write=False,
            )

    def commit_bounded_att_state(self, req_indices: torch.Tensor, accepted_steps: torch.Tensor) -> None:
        """按绝对位置寻址；接受后继续覆盖 ring，不维护额外 MTP 游标。"""
        pass

    # ========================================================================
    # 三、CPU Cache：仅搬运时间有序的大小页快照
    # ========================================================================

    @deepcopy_result
    @lru_cache(maxsize=None)
    def get_cpu_transfer_regions(self) -> tuple[TransferRegion, ...]:
        """校验缓存页对齐产出步长；无重叠历史的窗口不占 CPU 页，也不创建大小页池。"""
        args = get_env_start_args()
        # 大页是小页的整数倍；小页对齐产出步长后，大小页终点均不会落在未完成的组内。
        assert args.linear_att_hash_page_size % self.config.compress_stride == 0, (
            "state cache pages must align to the compressor stride"
        )
        # CPU 服务只查询 meta 布局，也要在忽略无重叠窗口之前拒绝不安全的恢复边界。
        if args.cpu_cache_token_page_size is not None:
            assert args.cpu_cache_token_page_size % self.config.compress_stride == 0, (
                "CPU cache pages must align to the compressor stride"
            )
        return self._get_transfer_regions(runtime=False)

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
            fields.append({"compressor": TensorField(tensor=pool.buffer[slot], axes=("layer", "window", "channel"))})
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
            {"compressor": TensorField(tensor=self.big_pages_pool.buffer[slot], axes=("layer", "window", "channel"))}
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
        remote_dp_bounded_state: WindowCompressorState,
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
        remote_dp_bounded_state: WindowCompressorState,
    ) -> None:
        """直接复制相同配置和 MTP 容量的物理 ring，不创建快照。"""
        assert self.config == remote_dp_bounded_state.config and self.mtp_step == remote_dp_bounded_state.mtp_step
        copy_bounded_att_tensors(
            copies=[
                (remote_dp_bounded_state.buffer[:, src], self.buffer[:, dst])
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
        return self._get_transfer_regions(runtime=True)

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

    def _get_transfer_regions(self, runtime: bool) -> tuple[TransferRegion, ...]:
        window = self.config.window_size if runtime else self.config.overlap_tokens
        if not window:
            return ()
        shape = (len(self.config.logical_layer_ids), window, 2 * self.config.candidate_dim)
        return (
            TransferRegion(
                field="compressor",
                byte_start=0,
                byte_end=prod(shape) * self.config.dtype.itemsize,
                dtype=self.config.dtype,
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
        copy_compressor_window(
            local=self.buffer[:, req_idx],
            external=external,
            state_token_len=state_token_len,
            write=write,
        )
