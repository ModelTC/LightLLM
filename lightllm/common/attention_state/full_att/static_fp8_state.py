from __future__ import annotations

from functools import lru_cache

from typing import TYPE_CHECKING, List
from collections.abc import Mapping
from dataclasses import dataclass
from math import prod

import torch

from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.log_utils import init_logger

from ..utils import deepcopy_result
from ..types import AxisMapping, TransferRegion
from .base_state import FullAttentionState
from .static_fp8_transfer import transfer_static_fp8_pages
from .dp_transfer import copy_gqa_tokens

if TYPE_CHECKING:
    from lightllm.common.attention_state.types import StateTransferLayout


logger = init_logger(name=__name__)


@dataclass(frozen=True)
class StaticFP8AttentionConfig:
    """静态 FP8 GQA 的逻辑布局；KV 固定为 FP8 编码，标定 Tensor 由后续流程绑定。"""

    # 外层顺序对应物理存储层，组内逻辑层共享该方案的全部字段；各组非空，逻辑层不能重复。
    # 例如 ((0, 4), (8,)) 分配两层，逻辑层 0/4 共用第一层的全部字段。
    # 分组不决定执行顺序或写入者；模型须让共享消费者跳过写入，避免覆盖生产层数据。
    logical_layer_groups: tuple[tuple[int, ...], ...]
    # TP 切分前单份 K 或 V 的 head 数，不将 K/V 相加，也不是 query head 数。
    global_head_num: int
    # 每个 K/V head 编码前的特征维度，量化位宽不改变此逻辑维度。
    head_dim: int
    # 所属模型 TP 组内的 rank，不是全局 rank 或 DP rank。
    tp_rank: int = 0
    # 所属模型 TP 组大小，用于确定分片或副本的访问关系。
    tp_world_size: int = 1
    # True 时各 rank 保存全部 heads；False 时切分，heads 少于 ranks 时相邻 ranks 复制同一 head。
    replicated: bool = False
    # 每层 K/V 各一个 scale，或每个本地 K/V head 各一个；不占用逐 token 页空间。
    scale_mode: str = "per_tensor"

    @property
    def head_num(self) -> int:
        """当前 TP rank 保存的单份 K 或 V 的 head 数。"""
        return self.global_head_num if self.replicated else max(1, self.global_head_num // self.tp_world_size)

    def __post_init__(self) -> None:
        assert isinstance(self.logical_layer_groups, tuple) and all(
            isinstance(group, tuple) and group for group in self.logical_layer_groups
        ), "layer groups must be a tuple of nonempty tuples"
        layers = [layer for group in self.logical_layer_groups for layer in group]
        assert all(isinstance(layer, int) and layer >= 0 for layer in layers), "logical layer IDs must be nonnegative integers"
        assert len(layers) == len(set(layers)), "a logical layer must belong to exactly one storage group"
        assert self.global_head_num > 0 and self.head_dim > 0 and 0 <= self.tp_rank < self.tp_world_size
        assert (
            self.replicated
            or self.global_head_num % self.tp_world_size == 0
            or self.tp_world_size % self.global_head_num == 0
        ), "KV heads require complete TP shards or replicas"
        assert self.scale_mode in ("per_tensor", "per_head"), "invalid static FP8 scale mode"


class StaticFP8FullAttentionState(FullAttentionState):
    """使用静态标定参数的 FP8 KV 存储，固定 scale 随 State 初始化，不逐 token 搬运。"""


    # ========================================================================
    # 一、资源定义与初始化
    # ========================================================================

    def __init__(
        self,
        layout_config: StaticFP8AttentionConfig,
        page_size: int = 1,
        device: str | torch.device = "cuda",
    ) -> None:
        """只准备静态 FP8 布局；标定参数由后续 bind_calibration 统一绑定。"""
        self.scales: torch.Tensor | None = None
        self.q_scales: torch.Tensor | None = None
        self.cpu_scales = None  # 实际存储初始化时准备计算 backend 使用的 CPU 标定值。
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
        self.k_storage: torch.Tensor | None = None
        self.v_storage: torch.Tensor | None = None
        self.storage_dtype = torch.uint8

    def bind_calibration(
        self,
        calibration: Mapping[str, object] | None,
        storage_layer_start: int,
    ) -> None:
        """在分配 KV 前绑定标定表中本 State 的物理层区间，并切出当前 TP 的 head。

        行起点由管理器按 main、draft0..draft35 的完整 State 集合确定，不是逻辑层号。
        同组逻辑层共用一行；State 内的行顺序与 logical_layer_groups 的物理层顺序一致。
        无文件时仅 per_tensor 可使用单位 scale；per_head 必须有 K/V 标定，Q 标定可选。
        未提供 Q 标定时由 attention backend 动态量化 Q。
        """
        assert self.k_storage is None and self.scales is None, "calibration must be bound once before storage allocation"
        config = self.config
        storage_layer_count = len(config.logical_layer_groups)
        if calibration is None:
            assert config.scale_mode == "per_tensor", "per-head FP8 requires a KV calibration file"
            scales = torch.ones(size=(storage_layer_count, 2), dtype=torch.float32)
            q_scales = None
        else:
            # 阶段一：检查文件编码和物理行覆盖范围。
            calibrated_layer_count = calibration["num_layers"]
            storage_layer_end = storage_layer_start + storage_layer_count
            assert 0 <= storage_layer_start < storage_layer_end <= calibrated_layer_count, (
                "calibration does not cover this State's physical layers"
            )
            assert calibration.get("quant_type", config.scale_mode) == config.scale_mode, (
                "calibration quantization mode does not match the State"
            )
            limits = torch.finfo(torch.float8_e4m3fn)
            assert calibration.get("qmin", limits.min) == limits.min and calibration.get("qmax", limits.max) == limits.max, (
                "calibration range does not match float8_e4m3fn"
            )

            def scale_tensor(values: Mapping[str, object], shape: tuple[int, ...]) -> torch.Tensor:
                tensor = torch.tensor(data=values["scales"], dtype=torch.float32)
                assert tuple(values["scales_shape"]) == tensor.shape == shape, "invalid calibration scale shape"
                assert bool((torch.isfinite(tensor) & (tensor > 0)).all()), "calibration scales must be finite and positive"
                return tensor[storage_layer_start:storage_layer_end]

            def shard(tensor: torch.Tensor, calibrated_heads: int) -> torch.Tensor:
                heads = config.head_num
                assert calibrated_heads > 0 and (heads * config.tp_world_size) % calibrated_heads == 0, (
                    "calibration heads do not cover the model's KV heads"
                )
                expanded = tensor.repeat_interleave(repeats=heads * config.tp_world_size // calibrated_heads, dim=-1)
                return expanded[:, config.tp_rank * heads : (config.tp_rank + 1) * heads].contiguous()

            # 阶段二：先按物理层取行，再分别切分 K、V、Q 的 head；标定 Tensor 暂留 CPU。
            if config.scale_mode == "per_head":
                calibrated_heads = calibration["num_head"]
                all_scales = scale_tensor(calibration, (calibrated_layer_count, 2 * calibrated_heads))
                k, v = all_scales.split(split_size=calibrated_heads, dim=-1)
                scales = torch.cat(tensors=(shard(k, calibrated_heads), shard(v, calibrated_heads)), dim=-1)
                q_config = calibration.get("q_calibration")
                q_scales = None
                if q_config is not None:
                    q_scales = shard(
                        scale_tensor(q_config, (calibrated_layer_count, q_config["num_head"])), q_config["num_head"],
                    )
            else:
                scales = scale_tensor(calibration, (calibrated_layer_count, 2))
                q_scales = None

        # 阶段三：绑定标定数值；后续 allocate_storage 将其迁移到实际设备。
        self.scales = scales
        self.q_scales = q_scales

    def get_state_type(self) -> str:
        """返回包含 per-tensor 或 per-head 标定方式的 FP8 方案标记。"""
        return "gqa_fp8_static_" + self.config.scale_mode

    def allocate_storage(self, token_capacity: int) -> None:
        """绑定 KV 与真实标定参数，并预备 FlashInfer 使用的 CPU scale。"""
        assert self.scales is not None, "static FP8 calibration must be bound before allocation"
        self.capacity = token_capacity
        shape = self._storage_shape()
        self.k_storage = torch.empty(size=shape, dtype=self.storage_dtype, device=self.device)
        self.v_storage = torch.empty(size=shape, dtype=self.storage_dtype, device=self.device)
        # 统一标定阶段保留 CPU 数值，实际分配时再迁移到 KV 所在设备。
        if self.k_storage.device.type != "meta":
            self.scales = self.scales.to(device=self.k_storage.device)
            if self.q_scales is not None:
                self.q_scales = self.q_scales.to(device=self.k_storage.device)
        # 提前准备 CPU scale，避免计算热路径同步设备。
        self.cpu_scales = self.scales.detach().cpu().numpy()

    def get_token_bytes(self) -> int:
        """由配置累计当前 TP rank 的 K/V 单 token 开销，无需分配存储。"""
        config = self.config
        return (
            len(config.logical_layer_groups) * 2 * config.head_num * config.head_dim
            * self.storage_dtype.itemsize
        )

    # ========================================================================
    # 二、Full attention：计算视图、KV 写入与完整 token 复制
    # ========================================================================

    def contains_logical_layer(self, logical_layer_id: int) -> bool:
        """判断指定逻辑层是否属于当前 State 的层映射，共享层同样返回 True。"""
        return logical_layer_id in self._logical_to_storage_layer

    def get_storage_layer_index(self, logical_layer_id: int) -> int:
        """按逻辑层号查询 KV 和标定参数共用的物理行；共享逻辑层返回同一下标。"""
        return self._logical_to_storage_layer[logical_layer_id]

    def get_layer_view(self, logical_layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """按逻辑层号返回 K、V 视图，head 数使用当前 TP 分片配置。"""
        storage_layer_index = self.get_storage_layer_index(logical_layer_id=logical_layer_id)
        return self.k_storage[storage_layer_index], self.v_storage[storage_layer_index]

    def write_full_att_kv(self, logical_layer_id: int, token_indices: torch.Tensor, kv: torch.Tensor) -> None:
        """按固定 per-tensor 或 per-head scale 量化输入 K/V，写入 FP8 编码。"""
        assert token_indices.is_cuda, "token indices must reside on CUDA"
        storage_layer_index = self.get_storage_layer_index(logical_layer_id=logical_layer_id)
        if not len(token_indices):
            return
        assert self.k_storage.is_cuda, "KV writes require CUDA storage"
        from lightllm.common.basemodel.triton_kernel.destindex_copy_kv_fp8 import destindex_copy_kv_fp8

        destindex_copy_kv_fp8(
            KV=kv,
            DestLoc=token_indices,
            scales=self.scales[storage_layer_index],
            Out=self.k_storage[storage_layer_index].view(torch.float8_e4m3fn),
            is_per_tensor_quant=self.config.scale_mode == "per_tensor",
            Out_v=self.v_storage[storage_layer_index].view(torch.float8_e4m3fn),
        )

    def copy_full_att_tokens(self, src_indices: torch.Tensor, dst_indices: torch.Tensor) -> None:
        """复制完整 token 数据；索引使用 Tensor，源与目标位置由调用方保证不重叠。"""
        assert src_indices.is_cuda and dst_indices.is_cuda, "token indices must reside on CUDA"
        assert src_indices.ndim == dst_indices.ndim == 1 and src_indices.numel() == dst_indices.numel(), (
            "token copies require equal one-dimensional index tensors"
        )
        assert src_indices.dtype in (torch.int32, torch.int64) and dst_indices.dtype in (torch.int32, torch.int64), (
            "token indices must be integer tensors"
        )
        assert self.k_storage.is_cuda, "token copies require CUDA storage"
        if not src_indices.numel():
            return
        from lightllm.common.basemodel.triton_kernel.kv_move import copy_kv_buffer_to_kv_buffer

        # 直接使用调用方提供的 CUDA 索引，保留原 dtype 和 stride。
        # 按字节直接从源槽位写到目标槽位，覆盖量化 scale 和 packed padding，不创建中间 KV。
        # 调用方保证源、目标集合不相交，且目标索引不重复，避免并行读写竞争。
        # TODO：将 FP8 K/V 原始编码 的同索引 token 复制融合为一次 kernel 发射，
        # 保持原始字节编码及各字段 stride，仍要求源目标不重叠、目标索引不重复。
        copy_kv_buffer_to_kv_buffer(
            src_mem_index=src_indices,
            dst_mem_index=dst_indices,
            kv_buffer=self.k_storage,
        )
        copy_kv_buffer_to_kv_buffer(
            src_mem_index=src_indices,
            dst_mem_index=dst_indices,
            kv_buffer=self.v_storage,
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
        transfer_static_fp8_pages(
            k_storage=self.k_storage,
            v_storage=self.v_storage,
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
        transfer_static_fp8_pages(
            k_storage=self.k_storage,
            v_storage=self.v_storage,
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
        remote_dp_full_att_state: StaticFP8FullAttentionState,
    ) -> None:
        """复制相同配置的 FP8 KV 编码；各 DP 使用相同的静态标定参数。"""
        assert type(remote_dp_full_att_state) is type(self) and remote_dp_full_att_state.config == self.config, (
            "DP copy requires the same encoding and equivalent TP state configurations"
        )
        assert src_token_indices.is_cuda and dst_token_indices.is_cuda, "DP token indices must reside on CUDA"
        assert src_token_indices.numel() == dst_token_indices.numel(), "source and destination token counts differ"
        copy_gqa_tokens(
            src_k_storage=remote_dp_full_att_state.k_storage,
            src_v_storage=remote_dp_full_att_state.v_storage,
            dst_k_storage=self.k_storage,
            dst_v_storage=self.v_storage,
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
        transfer_static_fp8_pages(
            k_storage=self.k_storage,
            v_storage=self.v_storage,
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
        transfer_static_fp8_pages(
            k_storage=self.k_storage,
            v_storage=self.v_storage,
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
        """依次排列 KV、动态量化等字段，描述全局页形状和本地 TP 分片。"""
        config = self.config
        byte_offset = 0
        regions = []
        logger.debug(
            "%s building transfer regions: TP=%d/%d, tokens_per_page=%d",
            self.state_name,
            config.tp_rank,
            config.tp_world_size,
            page_tokens,
        )
        # 阶段一：按本方案固定字段计算形状和 dtype，不分配 meta 存储。
        field_specs = (
            ("k", self.storage_dtype, config.head_dim, "dim"),
            ("v", self.storage_dtype, config.head_dim, "dim"),
        )
        for field_name, field_dtype, field_width, last_axis in field_specs:
            axes = ("layer", "token", "head", last_axis)
            local_shape = (len(config.logical_layer_groups), page_tokens, config.head_num, field_width)
            global_shape = list(local_shape)
            global_shape[axes.index("token")] = page_tokens
            head_axis = axes.index("head")
            # K/V 各自是独立字段：这里读取的是单个字段的本地 head 数。
            local_head_count = global_shape[head_axis]
            # 外部页保存该字段未做 TP 切分的完整 heads，TP 副本只占一份区域。
            # K/V 各生成一个 region 并分别累加字节数，因此这里不能再乘 2；动态 scale 同理。
            # 例如全局 H=8、TP=2：本地 k/v 各 4 heads，外部 k/v 区域各 8 heads。
            global_shape[head_axis] = config.global_head_num

            # 阶段二：确定本地 heads 的全局起点；同组副本只有首 rank 写出，所有副本都可读回。
            # replica_count 表示同一份 head 分片被多少个 ranks 复制：纯切分时为 1，全复制时为 TP。
            replica_count = config.tp_world_size * local_head_count // config.global_head_num
            # 将组内 rank 换算为去重后的分片编号，再定位当前字段在全局 head 轴上的起点。
            head_start = config.tp_rank // replica_count * local_head_count
            # 副本共享同一外部区域，只让组内第一个 rank 写出，避免重复写入；读取不受此标记限制。
            write_enabled = config.tp_rank % replica_count == 0
            axis_mappings = tuple(
                AxisMapping(
                    axis=axis,
                    local_start=0,
                    external_start=head_start if axis == "head" else 0,
                    length=dimension,
                )
                for axis, dimension in zip(axes, local_shape)
                if axis not in ("layer", "token")
            )

            # 阶段三：按字段 dtype 对齐起点，记录本字段区间，再推进下一个字段的字节偏移。
            element_bytes = field_dtype.itemsize
            byte_offset = (byte_offset + element_bytes - 1) // element_bytes * element_bytes
            field_region = TransferRegion(
                field=field_name,
                byte_start=byte_offset,
                byte_end=byte_offset + prod(global_shape) * element_bytes,
                dtype=field_dtype,
                axes=axes,
                shape=tuple(global_shape),
                axis_mappings=axis_mappings,
                write_enabled=write_enabled,
            )
            regions.append(field_region)
            byte_offset = field_region.byte_end
            logger.debug(
                "%s field %s: head_start=%d, heads=%d, replicas=%d, write_enabled=%s, bytes=[%d, %d)",
                self.state_name,
                field_name,
                head_start,
                local_head_count,
                replica_count,
                write_enabled,
                field_region.byte_start,
                field_region.byte_end,
            )
        logger.debug(
            "%s transfer regions built: fields=%d, total_bytes=%d",
            self.state_name,
            len(regions),
            byte_offset,
        )
        return tuple(regions)

    def _storage_shape(self) -> tuple[int, ...]:
        return (
            len(self.config.logical_layer_groups),
            self.capacity + self.page_size,
            self.config.head_num,
            self.config.head_dim,
        )
