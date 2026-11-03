from __future__ import annotations

from lightllm.common.attention_state.utils import weak_cached_method
from math import prod
from dataclasses import dataclass

import torch

from lightllm.utils.envs_utils import get_env_start_args

from ...utils import deepcopy_result, dtype_name
from ...types import AxisMapping, TransferRegion, TensorField
from ..base_state import FullAttentionState


@dataclass(frozen=True)
class MLAConfig:
    """普通 latent MLA 的逻辑配置，不包含 indexer 或 packed 物理字段。"""

    # 外层顺序对应物理存储层，组内逻辑层共享该方案的全部字段；各组非空，逻辑层不能重复。
    # 例如 ((0, 4), (8,)) 分配两层，逻辑层 0/4 共用第一层的全部字段。
    # 分组不决定执行顺序或写入者；模型须让共享消费者跳过写入，避免覆盖生产层数据。
    logical_layer_groups: tuple[tuple[int, ...], ...]
    # 编码前的完整 latent 宽度，包含模型需要的 RoPE 部分，不含 indexer 或 padding。
    latent_dim: int
    # 未压缩 latent 在运行态和存储中的浮点精度。
    latent_dtype: torch.dtype
    # 所属模型 TP 组内的 rank；各 rank 保存相同 latent，仅 rank 0 写出共享外部页。
    tp_rank: int = 0
    # 保存完整 latent 副本的 TP rank 数，不改变 latent 的维度或物理行宽。
    tp_world_size: int = 1

    def __post_init__(self) -> None:
        assert isinstance(self.logical_layer_groups, tuple) and all(
            isinstance(group, tuple) and group for group in self.logical_layer_groups
        ), "layer groups must be a tuple of nonempty tuples"
        layers = [layer for group in self.logical_layer_groups for layer in group]
        assert all(
            isinstance(layer, int) and layer >= 0 for layer in layers
        ), "logical layer IDs must be nonnegative integers"
        assert len(layers) == len(set(layers)), "a logical layer must belong to exactly one storage group"
        assert self.latent_dim > 0 and 0 <= self.tp_rank < self.tp_world_size


class MLAFullAttentionState(FullAttentionState):
    """存储复制到各 TP rank 的 MLA 压缩 latent，使用单个 KV 字段。"""

    # ========================================================================
    # 一、资源定义与初始化
    # ========================================================================

    storage_fields = ("storage",)

    def __init__(
        self,
        layout_config: MLAConfig,
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
        return "mla_" + dtype_name(dtype=self.config.latent_dtype)

    def get_transfer_fields(self):
        return {"kv": TensorField(self.storage, ("layer", "token", "head", "dim"))}

    def get_attention_format(self):
        from lightllm.common.basemodel.attention.contracts import MlaFormat

        return MlaFormat(self.config.latent_dtype, self.config.latent_dim, self.page_size)

    def get_storage_specs(self, token_capacity):
        from ...storage import TensorStorageSpec

        shape = (len(self.config.logical_layer_groups), token_capacity + self.page_size, 1, self.config.latent_dim)
        return (TensorStorageSpec("storage", shape, self.config.latent_dtype),)

    def get_token_bytes(self) -> int:
        """由配置计算当前 TP rank 的单 token latent 字节数，无需分配存储。"""
        config = self.config
        return len(config.logical_layer_groups) * config.latent_dim * config.latent_dtype.itemsize

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
        """各 rank 复制同一 latent，仅 rank 0 写出共享外部页，所有 rank 均可读回。"""
        config = self.config
        shape = (len(config.logical_layer_groups), page_tokens, 1, config.latent_dim)
        return (
            TransferRegion(
                field="kv",
                byte_start=0,
                byte_end=prod(shape) * config.latent_dtype.itemsize,
                dtype=config.latent_dtype,
                axes=("layer", "token", "head", "dim"),
                shape=shape,
                axis_mappings=(
                    AxisMapping(axis="head", local_start=0, external_start=0, length=1),
                    AxisMapping(axis="dim", local_start=0, external_start=0, length=config.latent_dim),
                ),
                write_enabled=config.tp_rank == 0,
            ),
        )
