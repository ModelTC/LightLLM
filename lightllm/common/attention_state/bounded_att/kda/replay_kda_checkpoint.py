"""KDA replay 的页布局和快照拷贝。

这里只处理 Conv/SSM 快照页，不读取 replay cursor 或 history。
"""

from __future__ import annotations

from math import prod
from typing import List

import torch

from lightllm.utils.log_utils import init_logger

from ...types import (
    AxisMapping,
    PageKind,
    StateTransferLayout,
    TensorField,
    TransferRegion,
)
from ..copy_kernel import copy_bounded_att_tensors
from ..transfer_utils import bounded_att_buffer_copies
from .kda_pages_pool import KDAPagesPool

logger = init_logger(__name__)


def build_transfer_regions(state) -> tuple[TransferRegion, ...]:
    """依次描述全局 Conv 和 SSM 区域，并标明当前 TP rank 的读写范围。"""
    # 阶段一：确定全局 Conv 形状；只传有效历史窗口，不包含 MTP 候选区和游标。
    config = state.config
    storage_layer_count = len(config.logical_layer_ids)
    conv_channels = 2 * config.global_k_heads * config.key_dim + config.global_v_heads * config.value_dim
    conv_shape = (storage_layer_count, conv_channels, config.conv_width)
    conv_byte_end = prod(conv_shape) * config.conv_dtype.itemsize
    regions = []
    logger.debug(
        "%s building transfer regions: TP=%d/%d, Conv shape=%s",
        state.state_name,
        config.tp_rank,
        config.tp_world_size,
        conv_shape,
    )

    # 阶段二：Conv 按 Q/K/V 拼接；三个映射共用同一块全局 Conv 区域，不能重复累计字节数。
    local_channel_start = 0
    global_channel_start = 0
    for name, local_head_count, global_head_count, head_dim in (
        ("Q", config.num_k_heads, config.global_k_heads, config.key_dim),
        ("K", config.num_k_heads, config.global_k_heads, config.key_dim),
        ("V", config.num_v_heads, config.global_v_heads, config.value_dim),
    ):
        # 配置已保证完整分片或副本。每组连续 rank 保存相同 heads，仅组内首 rank 写出。
        replica_count = config.tp_world_size * local_head_count // global_head_count
        head_start = config.tp_rank // replica_count * local_head_count
        write_enabled = config.tp_rank % replica_count == 0
        local_channel_count = local_head_count * head_dim
        target_channel_start = global_channel_start + head_start * head_dim
        conv_region = TransferRegion(
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
                AxisMapping(
                    axis="window",
                    local_start=0,
                    external_start=0,
                    length=config.conv_width,
                ),
            ),
            write_enabled=write_enabled,
        )
        regions.append(conv_region)
        logger.debug(
            "%s Conv %s: local_channel_start=%d, global_channel_start=%d, "
            "channels=%d, replicas=%d, write_enabled=%s",
            state.state_name,
            name,
            local_channel_start,
            target_channel_start,
            local_channel_count,
            replica_count,
            write_enabled,
        )
        local_channel_start += local_channel_count
        global_channel_start += global_head_count * head_dim

    # 阶段三：SSM 沿 V head 切分；接在 Conv 之后，按 SSM dtype 对齐字节起点。
    replica_count = config.tp_world_size * config.num_v_heads // config.global_v_heads
    head_start = config.tp_rank // replica_count * config.num_v_heads
    write_enabled = config.tp_rank % replica_count == 0
    ssm_shape = (
        storage_layer_count,
        config.global_v_heads,
        config.key_dim,
        config.value_dim,
    )
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
            AxisMapping(
                axis="head",
                local_start=0,
                external_start=head_start,
                length=config.num_v_heads,
            ),
            AxisMapping(
                axis="key_dim",
                local_start=0,
                external_start=0,
                length=config.key_dim,
            ),
            AxisMapping(
                axis="value_dim",
                local_start=0,
                external_start=0,
                length=config.value_dim,
            ),
        ),
        write_enabled=write_enabled,
    )
    regions.append(ssm_region)
    logger.debug(
        "%s transfer regions built: SSM head_start=%d, heads=%d, write_enabled=%s, SSM bytes=[%d, %d)",
        state.state_name,
        head_start,
        config.num_v_heads,
        write_enabled,
        ssm_region.byte_start,
        ssm_region.byte_end,
    )
    return tuple(regions)


def page_pool(state, page_kind):
    """根据大小页类型取得已启用的页池，不访问 allocator。"""
    assert page_kind in ("big", "small"), f"unknown page kind {page_kind}"
    pool = state.big_pages_pool if page_kind == "big" else state.small_pages_pool
    assert pool is not None, f"{page_kind} pages are disabled for {state.state_name}"
    return pool


def tensor_fields(conv: torch.Tensor, ssm: torch.Tensor):
    return {
        "conv": TensorField(tensor=conv, axes=("layer", "channel", "window")),
        "ssm": TensorField(tensor=ssm, axes=("layer", "head", "key_dim", "value_dim")),
    }


def write_pages_to_cpu(
    state,
    page_kind: PageKind,
    page_slot_ids: List[int],
    cpu_page_buffers: torch.Tensor,
    cpu_page_indices: List[int],
    layout: StateTransferLayout,
    cpu_page_ready_flags: list[bool] | None = None,
) -> None:
    """将指定页池槽位的快照写入 CPU 页。"""
    assert layout.state_name == state.state_name, "transfer layout does not describe this resource"
    assert 0 <= layout.byte_start < layout.byte_end, "invalid State byte interval"
    assert cpu_page_buffers.device.type == "cpu", "CPU cache buffer must reside on CPU"
    pool = page_pool(state, page_kind)
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
        page_fields.append(tensor_fields(pool.conv_state[slot], pool.ssm_state[slot]))
        page_indices.append(page)
    for copies in bounded_att_buffer_copies(
        page_fields=page_fields,
        page_buffers=cpu_page_buffers,
        layout=layout,
        page_indices=page_indices,
        write=True,
    ):
        copy_bounded_att_tensors(copies=copies)


def read_pages_from_cpu(
    state,
    cpu_page_buffers: torch.Tensor,
    cpu_page_indices: List[int],
    big_page_slot_ids: List[int],
    layout: StateTransferLayout,
) -> None:
    """读入指定大页槽位；请求运行态另行通过 restore_bounded_att_runtime_from_pages 恢复。"""
    assert layout.state_name == state.state_name, "transfer layout does not describe this resource"
    assert 0 <= layout.byte_start < layout.byte_end, "invalid State byte interval"
    assert cpu_page_buffers.device.type == "cpu", "CPU cache buffer must reside on CPU"
    pool = state.big_pages_pool
    assert isinstance(big_page_slot_ids, list), "slot indices must be a list"
    assert isinstance(cpu_page_indices, list), "CPU page indices must be a list"
    assert len(cpu_page_indices) == len(big_page_slot_ids), "CPU page and slot counts differ"
    # 本地目标：选定大页槽位的 Conv/SSM 快照，保留本 TP rank 的分片形状。
    pool_page_fields = [tensor_fields(pool.conv_state[slot], pool.ssm_state[slot]) for slot in big_page_slot_ids]
    # 外部来源：layout.regions 描述 CPU 全局字段和 TP 范围；write=False 将其读入本地页池。
    # 窗口保持快照顺序；请求 ring 的位置由后续 restore_bounded_att_runtime_from_pages 按长度恢复。
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


def copy_big_pages(
    state,
    src_page_ids: List[int],
    dst_page_ids: List[int],
    remote_dp_bounded_state,
) -> None:
    """复制同一 TP 分片的大页 Conv/SSM 快照，保留源页和双方请求运行态。"""
    assert isinstance(src_page_ids, list) and isinstance(dst_page_ids, list), "page indices must be lists"
    assert len(src_page_ids) == len(dst_page_ids), "source and destination page counts differ"
    assert (
        type(remote_dp_bounded_state) is type(state) and remote_dp_bounded_state.config == state.config
    ), "DP copy requires the same algorithm and equivalent TP state configurations"
    src_pool = remote_dp_bounded_state.big_pages_pool
    dst_pool = state.big_pages_pool
    assert src_pool is not None and dst_pool is not None, "DP big page copy requires both page pools"
    # 大页保存的是基准快照，与请求槽位及 MTP 候选区无关；页的引用和使用期限由调用方管理。
    # 页池均须在本进程注册；GPU 在当前流直接访问两端 CPU 存储，完成后才能归还页面。
    conv_copies, ssm_copies = [], []
    for src_page, dst_page in zip(src_page_ids, dst_page_ids):
        conv_copies.append((src_pool.conv_state[src_page], dst_pool.conv_state[dst_page]))
        ssm_copies.append((src_pool.ssm_state[src_page], dst_pool.ssm_state[dst_page]))
    copy_bounded_att_tensors(copies=conv_copies)
    copy_bounded_att_tensors(copies=ssm_copies)


def init_page_pools(
    state,
    big_page_capacity: int,
    small_page_capacity: int = 0,
    device: str | torch.device | int = "cpu",
    pin_memory: bool = True,
) -> None:
    """容量规划完成后创建本算法的大小页池；容量为零的页池保持 None。"""
    assert state.big_pages_pool is None and state.small_pages_pool is None, "page pools already initialized"
    assert big_page_capacity >= 0 and small_page_capacity >= 0, "page capacities must be nonnegative"
    state.big_pages_pool = (
        KDAPagesPool(
            size=big_page_capacity,
            config=state.config,
            device=device,
            pin_memory=pin_memory,
        )
        if big_page_capacity
        else None
    )
    state.small_pages_pool = (
        KDAPagesPool(
            size=small_page_capacity,
            config=state.config,
            device=device,
            pin_memory=pin_memory,
        )
        if small_page_capacity
        else None
    )


# ========================================================================
# 二、Bounded attention：计算视图、运行态重置、大小页与 MTP 提交
# ========================================================================
