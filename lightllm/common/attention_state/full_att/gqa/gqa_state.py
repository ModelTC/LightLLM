from __future__ import annotations

from lightllm.common.attention_state.utils import weak_cached_method
from math import prod
from dataclasses import dataclass

import torch

from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.log_utils import init_logger

from ...utils import deepcopy_result, dtype_name
from ...types import AxisMapping, TransferRegion, TensorField
from ..base_state import FullAttentionState


logger = init_logger(name=__name__)


@dataclass(frozen=True)
class GQAAttentionConfig:
    """普通 GQA（含 MHA/MQA）的逻辑 KV 布局；不描述量化编码或物理行 padding。"""

    # 外层顺序对应物理存储层，组内逻辑层共享该方案的全部字段；各组非空，逻辑层不能重复。
    # 例如 ((0, 4), (8,)) 分配两层，逻辑层 0/4 共用第一层的全部字段。
    # 分组不决定执行顺序或写入者；模型须让共享消费者跳过写入，避免覆盖生产层数据。
    logical_layer_groups: tuple[tuple[int, ...], ...]
    # TP 切分前单份 K 或 V 的 head 数，不将 K/V 相加，也不是 query head 数。
    global_head_num: int
    # 每个 K/V head 编码前的特征维度，量化位宽不改变此逻辑维度。
    head_dim: int
    # K/V 浮点存储精度。
    dtype: torch.dtype
    # 所属模型 TP 组内的 rank，不是全局 rank 或 DP rank。
    tp_rank: int = 0
    # 所属模型 TP 组大小，用于确定分片或副本的访问关系。
    tp_world_size: int = 1
    # True 时各 rank 保存全部 heads；False 时切分，heads 少于 ranks 时相邻 ranks 复制同一 head。
    replicated: bool = False

    @property
    def head_num(self) -> int:
        """当前 TP rank 保存的单份 K 或 V 的 head 数。"""
        return self.global_head_num if self.replicated else max(1, self.global_head_num // self.tp_world_size)

    def __post_init__(self) -> None:
        assert isinstance(self.logical_layer_groups, tuple) and all(
            isinstance(group, tuple) and group for group in self.logical_layer_groups
        ), "layer groups must be a tuple of nonempty tuples"
        layers = [layer for group in self.logical_layer_groups for layer in group]
        assert all(
            isinstance(layer, int) and layer >= 0 for layer in layers
        ), "logical layer IDs must be nonnegative integers"
        assert len(layers) == len(set(layers)), "a logical layer must belong to exactly one storage group"
        assert self.global_head_num > 0 and self.head_dim > 0 and 0 <= self.tp_rank < self.tp_world_size
        assert (
            self.replicated
            or self.global_head_num % self.tp_world_size == 0
            or self.tp_world_size % self.global_head_num == 0
        ), "KV heads require complete TP shards or replicas"


class GQAFullAttentionState(FullAttentionState):
    """GQA 的 full-attention KV 存储，MHA 和 MQA 均为其特例。

    TP 切分前，H_q 个 query head 分为 H_kv 组，每组共享一对 K/V head，
    要求 H_q 能被 H_kv 整除。MHA 对应 H_kv == H_q；MQA 对应 H_kv == 1；
    其余 GQA 对应 1 < H_kv < H_q。

    config.head_num 表示 TP 切分或复制后的本地 KV head 数。
    K/V 分别为 [layers, token_slots, local_kv_heads, head_dim]，两者独立分配。
    query 与 KV 的分组计算由 attention backend 处理，因此三种情况共用此 State。
    """

    storage_fields = ("k_storage", "v_storage")

    # 保留既有 K/V 编码的稳定传输布局标识。

    # ========================================================================
    # 一、资源定义与初始化
    # ========================================================================

    def __init__(
        self,
        layout_config: GQAAttentionConfig,
        page_size: int = 1,
        device: str | torch.device = "cuda",
    ) -> None:
        """只记录布局配置，KV 与 HOLD 页由后续统一容量初始化分配。"""
        self.config = layout_config
        self.storage_dtype = layout_config.dtype  # 实际 KV 编码类型；量化方案显式设置，区别于 scale 精度。
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

    def get_state_type(self) -> str:
        """返回 GQA KV 精度及全局 head 布局对应的存储方案标记。"""
        return f"gqa_{dtype_name(dtype=self.config.dtype)}_h{self.config.global_head_num}_d{self.config.head_dim}"

    def get_transfer_fields(self):
        return {
            name: TensorField(tensor, ("layer", "token", "head", "dim"))
            for name, tensor in (("k", self.k_storage), ("v", self.v_storage))
        }

    def get_attention_format(self):
        from lightllm.common.basemodel.attention.contracts import KVFormat

        config = self.config
        return KVFormat(
            "plain",
            self.storage_dtype,
            config.global_head_num,
            config.head_num,
            config.head_dim,
            self.page_size,
            config.tp_rank,
            config.tp_world_size,
            config.replicated,
        )

    def get_storage_specs(self, token_capacity):
        from ...storage import TensorStorageSpec

        shape = (
            len(self.config.logical_layer_groups),
            token_capacity + self.page_size,
            self.config.head_num,
            self.config.head_dim,
        )
        return tuple(TensorStorageSpec(name, shape, self.storage_dtype) for name in self.storage_fields)

    def get_token_bytes(self) -> int:
        """由配置累计当前 TP rank 的 K/V 单 token 开销，无需分配存储。"""
        config = self.config
        return len(config.logical_layer_groups) * 2 * config.head_num * config.head_dim * self.storage_dtype.itemsize

    # ========================================================================
    # 二、Full attention：计算视图、KV 写入与完整 token 复制
    # ========================================================================

    def contains_logical_layer(self, logical_layer_id: int) -> bool:
        """判断指定逻辑层是否属于当前 State 的层映射，共享层同样返回 True。"""
        return logical_layer_id in self._logical_to_storage_layer

    def get_storage_layer_index(self, logical_layer_id: int) -> int:
        """按逻辑层号查询存储及配套字段的物理行；共享逻辑层返回同一下标。"""
        return self._logical_to_storage_layer[logical_layer_id]

    def get_layer_view(self, logical_layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """按逻辑层号返回 K、V 视图，head 数使用当前 TP 分片配置。"""
        storage_layer_index = self.get_storage_layer_index(logical_layer_id=logical_layer_id)
        return self.k_storage[storage_layer_index], self.v_storage[storage_layer_index]

    # ========================================================================
    # 三、CPU Cache 数据交换
    # ========================================================================

    @deepcopy_result
    @weak_cached_method
    def get_cpu_transfer_regions(self) -> tuple[TransferRegion, ...]:
        """按全局 CPU 页大小返回字段区间与 TP 切片，作为 CPU 页容量计算依据。

        覆盖 KV、动态量化参数及 indexer 有效字段，不包含 GPU 行 padding 或静态标定参数。
        """
        page_tokens = get_env_start_args().cpu_cache_token_page_size
        return self._get_transfer_regions(page_tokens=page_tokens)

    # ========================================================================
    # 四、DP 迁移
    # ========================================================================

    # ========================================================================
    # 五、PD 分离传输
    # ========================================================================

    @deepcopy_result
    @weak_cached_method
    def get_pd_transfer_regions(self) -> tuple[TransferRegion, ...]:
        """按全局 PD 页大小返回字段与切片，交由管理器安排其在联合页中的位置。"""
        page_tokens = get_env_start_args().pd_kv_page_size
        return self._get_transfer_regions(page_tokens=page_tokens)

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
