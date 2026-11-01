from __future__ import annotations
from dataclasses import dataclass
from math import prod
from typing import Literal
import torch
from .storage import BoundedStorageSpec, RuntimeBuffer, SnapshotField
from ..types import FieldName, AxisMapping, TransferRegion
from ..utils import dtype_name


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


def conv_ssm_regions(config) -> tuple[TransferRegion, ...]:
    """依次描述全局 Conv 和 SSM 区域，并标明当前 TP rank 的读写范围。"""
    storage_layer_count = len(config.logical_layer_ids)
    conv_channels = 2 * config.global_k_heads * config.key_dim + config.global_v_heads * config.value_dim
    conv_shape = (storage_layer_count, conv_channels, config.conv_width)
    conv_byte_end = prod(conv_shape) * config.conv_dtype.itemsize
    regions = []
    local_channel_start = 0
    global_channel_start = 0
    for name, local_head_count, global_head_count, head_dim in (
        ("Q", config.num_k_heads, config.global_k_heads, config.key_dim),
        ("K", config.num_k_heads, config.global_k_heads, config.key_dim),
        ("V", config.num_v_heads, config.global_v_heads, config.value_dim),
    ):
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
                AxisMapping(axis="window", local_start=0, external_start=0, length=config.conv_width),
            ),
            write_enabled=write_enabled,
        )
        regions.append(conv_region)
        local_channel_start += local_channel_count
        global_channel_start += global_head_count * head_dim
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
    return tuple(regions)


def conv_ssm_spec(config, request_capacity, mtp_step, algorithm):
    layers = len(config.logical_layer_ids)
    buffers = (
        RuntimeBuffer(
            FieldName.CONV_STATE,
            (layers, request_capacity, config.conv_channels, config.conv_width + mtp_step),
            config.conv_dtype,
        ),
        RuntimeBuffer(
            FieldName.SSM_STATE,
            (layers, request_capacity * (mtp_step + 1), config.num_v_heads, config.key_dim, config.value_dim),
            config.ssm_dtype,
        ),
    )
    if mtp_step:
        buffers += (RuntimeBuffer(FieldName.MTP_STATE_INDEX, (request_capacity,), torch.int32),)
    regions = conv_ssm_regions(config)
    fields = (
        SnapshotField(
            FieldName.CONV,
            "conv_state",
            regions[0].axes,
            (layers, buffers[0].shape[2], config.conv_width),
            buffers[0].dtype,
        ),
        SnapshotField(FieldName.SSM, "ssm_state", regions[-1].axes, (layers, *buffers[1].shape[2:]), buffers[1].dtype),
    )
    return BoundedStorageSpec(
        f"{algorithm}_conv_{dtype_name(config.conv_dtype)}_ssm_{dtype_name(config.ssm_dtype)}",
        config.logical_layer_ids,
        buffers,
        fields,
        regions,
        regions,
    )


def window_regions(config, name, window, width, dtype):
    if not window:
        return ()
    shape = (len(config.logical_layer_ids), window, width)
    return (
        TransferRegion(
            field=name,
            byte_start=0,
            byte_end=prod(shape) * dtype.itemsize,
            dtype=dtype,
            axes=("layer", "window", "channel"),
            shape=shape,
            axis_mappings=(),
            write_enabled=config.tp_rank == 0,
        ),
    )


def sliding_window_spec(config, request_capacity, mtp_step):
    page_tokens = config.page_token_size
    capacity = (config.window_size + 2 * mtp_step + page_tokens - 1) // page_tokens * page_tokens
    buffers = (
        RuntimeBuffer(
            FieldName.SWA_BUFFER,
            (
                len(config.logical_layer_ids),
                request_capacity,
                capacity,
                config.get_packed_kv_bytes(page_tokens) // page_tokens,
            ),
            torch.uint8,
        ),
    )
    regions = window_regions(config, "swa", config.window_size, config.kv_bytes, torch.uint8)
    return BoundedStorageSpec(
        f"swa_w{config.window_size}_fp8_group64_e8m0",
        config.logical_layer_ids,
        buffers,
        (SnapshotField(FieldName.SWA, "buffer", regions[0].axes, regions[0].shape, regions[0].dtype),),
        regions,
        regions,
    )


def compressor_spec(config, request_capacity, mtp_step):
    precision = {torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16"}[config.dtype]
    buffers = (
        RuntimeBuffer(
            FieldName.BUFFER,
            (len(config.logical_layer_ids), request_capacity, config.window_size + mtp_step, 2 * config.candidate_dim),
            config.dtype,
        ),
    )
    cpu_regions = window_regions(config, "compressor", config.overlap_tokens, buffers[0].shape[-1], buffers[0].dtype)
    pd_regions = window_regions(config, "compressor", config.window_size, buffers[0].shape[-1], buffers[0].dtype)
    fields = tuple(SnapshotField(FieldName.COMPRESSOR, "buffer", r.axes, r.shape, r.dtype) for r in cpu_regions)
    return BoundedStorageSpec(
        f"compressor_{config.target}_w{config.window_size}_s{config.compress_stride}_{precision}",
        config.logical_layer_ids,
        buffers,
        fields,
        cpu_regions,
        pd_regions,
    )
