"""KDA replay 的 bounded attention 状态。

Conv、canonical SSM、replay history、页布局和已接受快照都在本类。
一层 decode 重放在 replay_kda_decode。
"""

from __future__ import annotations

from functools import lru_cache
from math import prod

import torch

from lightllm.common.basemodel.triton_kernel.linear_att.replayssm_cursor import advance_cursor
from lightllm.common.basemodel.triton_kernel.linear_att.replayssm_fold import fold_prefix
from lightllm.common.basemodel.triton_kernel.linear_att.ssm_acceptance import commit_ssm_prefix
from lightllm.utils.log_utils import init_logger

from ...types import AxisMapping, PageKind, StateTransferLayout, TensorField, TransferRegion
from ...utils import deepcopy_result, dtype_name
from ..base_state import BoundedAttentionState
from ..conv_ssm_types import ConvSSMStateConfig
from ..copy_kernel import copy_bounded_att_tensors
from ..transfer_utils import bounded_att_buffer_copies
from .kda_pages_pool import KDAPagesPool

logger = init_logger(__name__)


class ReplayKDAState(BoundedAttentionState):
    """KDA replay 运行态。Conv、单槽 SSM 和 history 都在本类，不沿用 native 的 MTP 槽。"""

    # ========================================================================
    # 一、资源定义与初始化
    # ========================================================================

    def __init__(
        self,
        config: ConvSSMStateConfig,
        request_capacity: int,
        mtp_step: int = 0,
        device: str | torch.device = "cuda",
        ssm_state_mode: str = "replay",
        ssm_replay_config: dict | None = None,
        gate_lower_bound: float = -5.0,
    ) -> None:
        """创建单槽 canonical SSM，并挂上只属于当前请求的 replay history。"""
        assert ssm_state_mode == "replay", f"ReplayKDAState only implements replay mode, got {ssm_state_mode}"
        assert isinstance(config, ConvSSMStateConfig), "config must be ConvSSMStateConfig"
        self.config = config
        self.state_name = None
        self.request_capacity = request_capacity
        self.mtp_step = mtp_step
        self.mtp_enabled = mtp_step > 0
        self.gate_lower_bound = gate_lower_bound
        self._algorithm = "kda"
        self._logical_to_storage_layer = {layer_id: index for index, layer_id in enumerate(config.logical_layer_ids)}
        layer_count = len(config.logical_layer_ids)
        self.conv_state = torch.zeros(
            (
                layer_count,
                request_capacity,
                config.conv_channels,
                config.conv_width + mtp_step,
            ),
            dtype=config.conv_dtype,
            device=device,
        )
        self.ssm_state = torch.zeros(
            (
                layer_count,
                request_capacity,
                config.num_v_heads,
                config.key_dim,
                config.value_dim,
            ),
            dtype=config.ssm_dtype,
            device=device,
        )
        self.mtp_state_index = torch.zeros(request_capacity, dtype=torch.int32, device=device)
        self.big_pages_pool = None
        self.small_pages_pool = None
        replay_config = dict(ssm_replay_config or {})
        capacity = int(replay_config.get("cache_len", 16))
        # 宽度为 1 逐 token 重放。开了 MTP 时默认先投影整个验证窗口。
        compute_mode = replay_config.get("compute_mode", "precompute" if mtp_step > 0 else "inline")
        verify_width = mtp_step + 1
        # 连续前缀挂在上面的 canonical SSM 上，不再另分配状态。
        # capacity 是 2 的幂，并且不小于一次验证窗口。
        assert self.ssm_state.dtype in (torch.float32, torch.bfloat16), "KDA replay requires FP32 or BF16 SSM state"
        assert verify_width > 0 and capacity >= max(4, verify_width) and capacity & (capacity - 1) == 0
        assert compute_mode in ("inline", "precompute")
        self.kda = True
        self.capacity = capacity
        self.verify_width = verify_width
        self.compute_mode = compute_mode
        layers, slots, hv, k, v = self.ssm_state.shape
        self.num_key_heads = config.num_k_heads
        assert self.num_key_heads > 0 and hv % self.num_key_heads == 0
        # 最后一个 slot 是 HOLD，replay 写零并且不记 history。
        self.hold_req_id = slots - 1
        # cursor 是已接受、尚未折回的长度。各层只读，最后一层返回后才按整窗推进。
        self.cursors = torch.zeros(slots, dtype=torch.int32, device=device)
        # 按 launch 行和 key head 同步。每个请求槽留 mtp_step + 6 行，盖住验证窗口和重复的 HOLD。
        self.tile_sync_mark = torch.zeros(
            (slots * (mtp_step + 6), self.num_key_heads), dtype=torch.int32, device=device
        )
        # 归一化 k 有范围保证，bf16 激活用 fp16 保留更多尾数；d 无此保证，使用 fp32。
        # 折回不用这份 d，而用下面的原始 k/v，避免把重建误差写进 SSM。g、beta 保持 fp32。
        # KDA 的 g 按 key 通道存。
        reconstruction_dtype = torch.float16 if config.conv_dtype == torch.bfloat16 else torch.float32
        key_shape = (layers, slots, self.num_key_heads, capacity, k)
        self.keys = torch.empty(key_shape, dtype=reconstruction_dtype, device=device)
        self.deltas = torch.empty((layers, slots, hv, capacity, v), dtype=torch.float32, device=device)
        self.gates = torch.empty((layers, slots, hv, capacity, k), dtype=torch.float32, device=device)
        self.raw_keys = torch.empty(key_shape, dtype=config.conv_dtype, device=device)
        self.raw_values = torch.empty((layers, slots, hv, capacity, v), dtype=config.conv_dtype, device=device)
        self.betas = torch.empty((layers, slots, hv, capacity), dtype=torch.float32, device=device)

    def get_state_type(self) -> str:
        """返回算法名和 Conv/SSM 存储精度。"""
        return (
            f"{self._algorithm}_conv_{dtype_name(dtype=self.config.conv_dtype)}"
            f"_ssm_{dtype_name(dtype=self.config.ssm_dtype)}"
        )

    def initialize_page_pools(
        self,
        big_page_capacity: int,
        small_page_capacity: int = 0,
        device: str | torch.device | int = "cpu",
        pin_memory: bool = True,
    ) -> None:
        """容量规划完成后创建本算法的大小页池；容量为零的页池保持 None。"""
        assert self.big_pages_pool is None and self.small_pages_pool is None, "page pools already initialized"
        assert big_page_capacity >= 0 and small_page_capacity >= 0, "page capacities must be nonnegative"
        self.big_pages_pool = (
            KDAPagesPool(
                size=big_page_capacity,
                config=self.config,
                device=device,
                pin_memory=pin_memory,
            )
            if big_page_capacity
            else None
        )
        self.small_pages_pool = (
            KDAPagesPool(
                size=small_page_capacity,
                config=self.config,
                device=device,
                pin_memory=pin_memory,
            )
            if small_page_capacity
            else None
        )

    # ========================================================================
    # 二、计算视图、运行态重置、已接受快照与 MTP 提交
    # ========================================================================

    def contains_logical_layer(self, logical_layer_id: int) -> bool:
        """判断指定逻辑层是否属于当前 State 的层映射。"""
        return logical_layer_id in self._logical_to_storage_layer

    def storage_layer(self, logical_layer_id: int) -> int:
        """返回逻辑层在 history 和 canonical state 里的存储下标。"""

        return self._logical_to_storage_layer[logical_layer_id]

    def get_layer_view(self, logical_layer_id: int):
        index = self.storage_layer(logical_layer_id)
        return self.conv_state[index], self.ssm_state[index]

    def reset_bounded_att_state(self, req_indices: list[int]) -> None:
        """清零请求的 Conv、单槽 SSM、history 和 MTP 游标。"""
        assert isinstance(req_indices, list), "request indices must be a list"
        for req_idx in req_indices:
            self.conv_state[:, req_idx].zero_()
            self.ssm_state[:, req_idx].zero_()
            self.cursors[req_idx].zero_()
            if self.mtp_enabled:
                self.mtp_state_index[req_idx].fill_(0)

    def save_bounded_att_runtime_to_pages(
        self,
        req_indices: list[int],
        state_token_lens: list[int],
        page_kind,
        page_slot_ids: list[int],
    ) -> None:
        """把 prefill 基准 Conv 窗口和 canonical SSM 写入页。history 与 MTP 候选区不参与。"""
        assert isinstance(req_indices, list) and isinstance(page_slot_ids, list), "indices must be lists"
        assert len(req_indices) == len(state_token_lens) == len(page_slot_ids), "request, length and page counts differ"
        pool = self._pool(page_kind)
        width = self.config.conv_width
        conv_copies, ssm_copies = [], []
        for req_idx, slot_id in zip(req_indices, page_slot_ids):
            conv_copies.append((self.conv_state[:, req_idx, :, :width], pool.conv_state[slot_id]))
            ssm_copies.append((self.ssm_state[:, req_idx], pool.ssm_state[slot_id]))
        copy_bounded_att_tensors(copies=conv_copies)
        copy_bounded_att_tensors(copies=ssm_copies)

    def restore_bounded_att_runtime_from_pages(
        self,
        req_indices: list[int],
        state_token_lens: list[int],
        page_kind,
        page_slot_ids: list[int],
    ) -> None:
        """从页恢复基准 Conv/SSM。使目标槽的旧 history 失效。"""
        assert isinstance(req_indices, list) and isinstance(page_slot_ids, list), "indices must be lists"
        assert len(req_indices) == len(state_token_lens) == len(page_slot_ids), "request, length and page counts differ"
        pool = self._pool(page_kind)
        width = self.config.conv_width
        conv_copies, ssm_copies = [], []
        for req_idx, slot_id in zip(req_indices, page_slot_ids):
            conv_copies.append((pool.conv_state[slot_id], self.conv_state[:, req_idx, :, :width]))
            ssm_copies.append((pool.ssm_state[slot_id], self.ssm_state[:, req_idx]))
        copy_bounded_att_tensors(copies=conv_copies)
        copy_bounded_att_tensors(copies=ssm_copies)
        for req_idx in req_indices:
            self.cursors[req_idx].zero_()
            if self.mtp_enabled:
                self.mtp_state_index[req_idx].fill_(0)

    def commit_bounded_att_state(
        self,
        req_indices: torch.Tensor,
        accepted_steps: torch.Tensor,
        req_verify_width: torch.Tensor,
    ) -> None:
        """MTP 验证后写入索引，并从整窗全接受后的 cursor 上剪掉未接受尾部。"""
        if not self.mtp_enabled:
            return
        commit_ssm_prefix(self.mtp_state_index, req_indices, accepted_steps, req_verify_width, self.cursors)

    # ========================================================================
    # 三、CPU Cache 数据交换
    # ========================================================================

    @deepcopy_result
    @lru_cache(maxsize=None)
    def get_cpu_transfer_regions(self) -> tuple[TransferRegion, ...]:
        """返回已接受 Conv/SSM 快照的字段区间。不含 history 或 MTP 候选区。"""
        return self._transfer_regions()

    def write_state_pages_to_cpu(
        self,
        page_kind: PageKind,
        page_slot_ids: list[int],
        cpu_page_buffers: torch.Tensor,
        cpu_page_indices: list[int],
        layout: StateTransferLayout,
        cpu_page_ready_flags: list[bool] | None = None,
    ) -> None:
        """将指定页池槽位的快照写入 CPU 页。"""
        assert layout.state_name == self.state_name, "transfer layout does not describe this resource"
        assert 0 <= layout.byte_start < layout.byte_end, "invalid State byte interval"
        assert cpu_page_buffers.device.type == "cpu", "CPU cache buffer must reside on CPU"
        pool = self._pool(page_kind)
        assert isinstance(page_slot_ids, list), "slot indices must be a list"
        assert isinstance(cpu_page_indices, list), "CPU page indices must be a list"
        assert len(page_slot_ids) == len(cpu_page_indices), "CPU page and slot counts differ"
        if cpu_page_ready_flags is not None:
            assert len(cpu_page_ready_flags) == len(cpu_page_indices), "CPU page and ready counts differ"
        page_fields, page_indices = [], []
        for index, (slot, page) in enumerate(zip(page_slot_ids, cpu_page_indices)):
            # ready 页不参与复制；同一传输区域的所有页面组成一个同形状批次。
            if cpu_page_ready_flags is not None and cpu_page_ready_flags[index]:
                continue
            page_fields.append(self._tensor_fields(pool.conv_state[slot], pool.ssm_state[slot]))
            page_indices.append(page)
        for copies in bounded_att_buffer_copies(
            page_fields=page_fields,
            page_buffers=cpu_page_buffers,
            layout=layout,
            page_indices=page_indices,
            write=True,
        ):
            copy_bounded_att_tensors(copies=copies)

    def read_state_pages_from_cpu(
        self,
        cpu_page_buffers: torch.Tensor,
        cpu_page_indices: list[int],
        big_page_slot_ids: list[int],
        layout: StateTransferLayout,
    ) -> None:
        """读入指定大页槽位。请求运行态由 restore_bounded_att_runtime_from_pages 另行恢复。"""
        assert layout.state_name == self.state_name, "transfer layout does not describe this resource"
        assert 0 <= layout.byte_start < layout.byte_end, "invalid State byte interval"
        assert cpu_page_buffers.device.type == "cpu", "CPU cache buffer must reside on CPU"
        pool = self.big_pages_pool
        assert isinstance(big_page_slot_ids, list), "slot indices must be a list"
        assert isinstance(cpu_page_indices, list), "CPU page indices must be a list"
        assert len(cpu_page_indices) == len(big_page_slot_ids), "CPU page and slot counts differ"
        # 本地目标保留本 TP rank 的分片形状。layout.regions 描述 CPU 全局字段。
        pool_page_fields = [
            self._tensor_fields(pool.conv_state[slot], pool.ssm_state[slot]) for slot in big_page_slot_ids
        ]
        for copies in bounded_att_buffer_copies(
            page_fields=pool_page_fields,
            page_buffers=cpu_page_buffers,
            layout=layout,
            page_indices=cpu_page_indices,
            write=False,
        ):
            copy_bounded_att_tensors(copies=copies)

    # ========================================================================
    # 四、DP 迁移：大页快照与基准运行态
    # ========================================================================

    def copy_big_pages_from_dp(self, src_page_ids: list[int], dst_page_ids: list[int], remote_dp_bounded_state) -> None:
        """复制同一 TP 分片的大页 Conv/SSM 快照，保留源页和双方请求运行态。"""
        assert isinstance(src_page_ids, list) and isinstance(dst_page_ids, list), "page indices must be lists"
        assert len(src_page_ids) == len(dst_page_ids), "source and destination page counts differ"
        assert (
            type(remote_dp_bounded_state) is type(self) and remote_dp_bounded_state.config == self.config
        ), "DP copy requires the same algorithm and equivalent TP state configurations"
        src_pool = remote_dp_bounded_state.big_pages_pool
        dst_pool = self.big_pages_pool
        assert src_pool is not None and dst_pool is not None, "DP big page copy requires both page pools"
        # 大页是基准快照，与请求槽位无关。页池须在本进程注册，当前流访问完才能归还页面。
        conv_copies, ssm_copies = [], []
        for src_page, dst_page in zip(src_page_ids, dst_page_ids):
            conv_copies.append((src_pool.conv_state[src_page], dst_pool.conv_state[dst_page]))
            ssm_copies.append((src_pool.ssm_state[src_page], dst_pool.ssm_state[dst_page]))
        copy_bounded_att_tensors(copies=conv_copies)
        copy_bounded_att_tensors(copies=ssm_copies)

    def copy_runtime_state_from_dp(
        self,
        src_req_indices: list[int],
        dst_req_indices: list[int],
        remote_dp_bounded_state: "ReplayKDAState",
    ) -> None:
        """复制 prefill 基准 Conv/SSM 到目标单槽。使目标槽的旧 history 失效。"""
        assert isinstance(src_req_indices, list) and isinstance(dst_req_indices, list), "request indices must be lists"
        assert len(src_req_indices) == len(dst_req_indices), "source and destination request counts differ"
        assert (
            type(remote_dp_bounded_state) is type(self) and remote_dp_bounded_state.config == self.config
        ), "DP copy requires the same algorithm and equivalent TP state configurations"
        width = self.config.conv_width
        conv_copies, ssm_copies = [], []
        for src_req, dst_req in zip(src_req_indices, dst_req_indices):
            conv_copies.append(
                (
                    remote_dp_bounded_state.conv_state[:, src_req, :, :width],
                    self.conv_state[:, dst_req, :, :width],
                )
            )
            ssm_copies.append((remote_dp_bounded_state.ssm_state[:, src_req], self.ssm_state[:, dst_req]))
        copy_bounded_att_tensors(copies=conv_copies)
        copy_bounded_att_tensors(copies=ssm_copies)
        for dst_req in dst_req_indices:
            self.cursors[dst_req].zero_()
            if self.mtp_enabled:
                self.mtp_state_index[dst_req].fill_(0)

    # ========================================================================
    # 五、PD 分离传输
    # ========================================================================

    @deepcopy_result
    @lru_cache(maxsize=None)
    def get_pd_transfer_regions(self) -> tuple[TransferRegion, ...]:
        """返回 PD 传输字段与切片，交由管理器安排其在联合页中的位置。"""
        return self._transfer_regions()

    def write_to_pd_page(
        self,
        req_idx: int,
        pd_page_buffers: torch.Tensor,
        pd_page_index: int,
        state_token_len: int,
        layout: StateTransferLayout,
    ) -> None:
        """写出 prefill 基准 Conv 窗口和 canonical SSM。源 history 不参与。"""
        assert layout.state_name == self.state_name, "transfer layout does not describe this resource"
        assert 0 <= layout.byte_start < layout.byte_end, "invalid State byte interval"
        fields = self._tensor_fields(
            self.conv_state[:, req_idx, :, : self.config.conv_width],
            self.ssm_state[:, req_idx],
        )
        for copies in bounded_att_buffer_copies(
            page_fields=[fields],
            page_buffers=pd_page_buffers,
            layout=layout,
            page_indices=[pd_page_index],
            write=True,
        ):
            copy_bounded_att_tensors(copies=copies)

    def read_from_pd_page(
        self,
        pd_page_buffers: torch.Tensor,
        req_idx: int,
        pd_page_index: int,
        state_token_len: int,
        layout: StateTransferLayout,
    ) -> None:
        """读入基准 Conv/SSM。使目标槽的旧 history 失效。"""
        assert layout.state_name == self.state_name, "transfer layout does not describe this resource"
        assert 0 <= layout.byte_start < layout.byte_end, "invalid State byte interval"
        fields = self._tensor_fields(
            self.conv_state[:, req_idx, :, : self.config.conv_width],
            self.ssm_state[:, req_idx],
        )
        for copies in bounded_att_buffer_copies(
            page_fields=[fields],
            page_buffers=pd_page_buffers,
            layout=layout,
            page_indices=[pd_page_index],
            write=False,
        ):
            copy_bounded_att_tensors(copies=copies)
        self.cursors[req_idx].zero_()
        if self.mtp_enabled:
            self.mtp_state_index[req_idx].fill_(0)

    # ========================================================================
    # 六、Replay history
    #
    # cursor 是已接受、尚未折回的长度，各层只读。
    # 最后一层返回后按整窗全接受推进：放不下时 cursor 变成窗口长度，否则加上窗口长度。
    # MTP 验证后再减去没接受的尾部。宽度为 1 时窗口就是这一个 token，没有尾部可剪。
    # 前缀放不下时，折回发生在 replay kernel 内。fold_accepted_prefix 把已接受前缀写回 canonical SSM。
    # ========================================================================

    def finish_decode_token(self, logical_layer_id: int, reqs: torch.Tensor, cu_seqlens: torch.Tensor) -> None:
        """最后一层按整窗全接受推进 cursor。非最后一层直接返回。"""

        layers = () if self.config is None else self.config.logical_layer_ids
        if not layers or logical_layer_id != layers[-1] or reqs.numel() == 0:
            return
        advance_cursor(self, reqs, cu_seqlens)

    def fold_accepted_prefix(self, reqs: torch.Tensor) -> None:
        """用原始 k/v 把 cursor 覆盖的已接受前缀折进 canonical SSM，并清掉这些请求的 cursor。"""

        if reqs.numel() == 0:
            return
        fold_prefix(self, reqs)
        self.cursors.index_fill_(0, reqs.long(), 0)

    # ========================================================================
    # 七、私有辅助实现
    # ========================================================================

    def _transfer_regions(self) -> tuple[TransferRegion, ...]:
        """依次描述全局 Conv 和 SSM 区域，并标明当前 TP rank 的读写范围。"""
        # 只传有效历史窗口，不包含 MTP 候选区和游标。
        config = self.config
        storage_layer_count = len(config.logical_layer_ids)
        conv_channels = 2 * config.global_k_heads * config.key_dim + config.global_v_heads * config.value_dim
        conv_shape = (storage_layer_count, conv_channels, config.conv_width)
        conv_byte_end = prod(conv_shape) * config.conv_dtype.itemsize
        regions = []
        logger.debug(
            "%s building transfer regions: TP=%d/%d, Conv shape=%s",
            self.state_name,
            config.tp_rank,
            config.tp_world_size,
            conv_shape,
        )

        # Conv 按 Q/K/V 拼接。三个映射共用同一块全局 Conv 区域，不能重复累计字节数。
        local_channel_start = 0
        global_channel_start = 0
        for name, local_head_count, global_head_count, head_dim in (
            ("Q", config.num_k_heads, config.global_k_heads, config.key_dim),
            ("K", config.num_k_heads, config.global_k_heads, config.key_dim),
            ("V", config.num_v_heads, config.global_v_heads, config.value_dim),
        ):
            # 每组连续 rank 保存相同 heads，仅组内首 rank 写出。
            replica_count = config.tp_world_size * local_head_count // global_head_count
            head_start = config.tp_rank // replica_count * local_head_count
            write_enabled = config.tp_rank % replica_count == 0
            local_channel_count = local_head_count * head_dim
            target_channel_start = global_channel_start + head_start * head_dim
            regions.append(
                TransferRegion(
                    field="conv",
                    byte_start=0,
                    byte_end=conv_byte_end,
                    dtype=config.conv_dtype,
                    axes=("layer", "channel", "window"),
                    shape=conv_shape,
                    axis_mappings=(
                        AxisMapping(
                            axis="channel",
                            local_start=local_channel_start,
                            external_start=target_channel_start,
                            length=local_channel_count,
                        ),
                        AxisMapping(axis="window", local_start=0, external_start=0, length=config.conv_width),
                    ),
                    write_enabled=write_enabled,
                )
            )
            logger.debug(
                "%s Conv %s: local_channel_start=%d, global_channel_start=%d, "
                "channels=%d, replicas=%d, write_enabled=%s",
                self.state_name,
                name,
                local_channel_start,
                target_channel_start,
                local_channel_count,
                replica_count,
                write_enabled,
            )
            local_channel_start += local_channel_count
            global_channel_start += global_head_count * head_dim

        # SSM 沿 V head 切分，接在 Conv 之后，并按 SSM dtype 对齐字节起点。
        replica_count = config.tp_world_size * config.num_v_heads // config.global_v_heads
        head_start = config.tp_rank // replica_count * config.num_v_heads
        write_enabled = config.tp_rank % replica_count == 0
        ssm_shape = (storage_layer_count, config.global_v_heads, config.key_dim, config.value_dim)
        element_bytes = config.ssm_dtype.itemsize
        ssm_byte_start = (conv_byte_end + element_bytes - 1) // element_bytes * element_bytes
        ssm_region = TransferRegion(
            field="ssm",
            byte_start=ssm_byte_start,
            byte_end=ssm_byte_start + prod(ssm_shape) * element_bytes,
            dtype=config.ssm_dtype,
            axes=("layer", "head", "key_dim", "value_dim"),
            shape=ssm_shape,
            axis_mappings=(
                AxisMapping(axis="head", local_start=0, external_start=head_start, length=config.num_v_heads),
                AxisMapping(axis="key_dim", local_start=0, external_start=0, length=config.key_dim),
                AxisMapping(axis="value_dim", local_start=0, external_start=0, length=config.value_dim),
            ),
            write_enabled=write_enabled,
        )
        regions.append(ssm_region)
        logger.debug(
            "%s transfer regions built: SSM head_start=%d, heads=%d, write_enabled=%s, SSM bytes=[%d, %d)",
            self.state_name,
            head_start,
            config.num_v_heads,
            write_enabled,
            ssm_region.byte_start,
            ssm_region.byte_end,
        )
        return tuple(regions)

    def _pool(self, page_kind: PageKind):
        """根据大小页类型取得已启用的页池，不访问 allocator。"""
        assert page_kind in ("big", "small"), f"unknown page kind {page_kind}"
        pool = self.big_pages_pool if page_kind == "big" else self.small_pages_pool
        assert pool is not None, f"{page_kind} pages are disabled for {self.state_name}"
        return pool

    @staticmethod
    def _tensor_fields(conv: torch.Tensor, ssm: torch.Tensor):
        """描述一页里的 Conv 和 SSM 字段。"""
        return {
            "conv": TensorField(tensor=conv, axes=("layer", "channel", "window")),
            "ssm": TensorField(tensor=ssm, axes=("layer", "head", "key_dim", "value_dim")),
        }
