from __future__ import annotations

from lightllm.common.attention_state.utils import weak_cached_method
from math import prod

import torch

from lightllm.utils.log_utils import init_logger

from ..utils import deepcopy_result, dtype_name
from ..types import AxisMapping, TensorField, TransferRegion
from .base_state import BoundedAttentionState
from .conv_ssm_types import ConvSSMStateConfig
from .copy_plan import BatchedStateView


logger = init_logger(__name__)


class ConvSSMState(BoundedAttentionState):

    algorithm: str
    runtime_storage_fields = ("conv_state", "ssm_state", "mtp_state_index")

    @property
    def resource_kind(self):
        return self.algorithm

    # ========================================================================
    # 一、资源定义与初始化
    # ========================================================================

    def __init__(
        self,
        config: ConvSSMStateConfig,
        request_capacity: int,
        mtp_step: int = 0,
    ) -> None:
        """按请求容量创建 Conv/SSM 运行态和 MTP 游标；大小页池延后初始化。"""
        assert isinstance(config, ConvSSMStateConfig), "config must be ConvSSMStateConfig"
        self.config = config
        self.state_name: str | None = None  # 由 AttentionStateManager.register_state() 绑定时设置。
        self.request_capacity = request_capacity  # 包含预留的 HOLD 请求槽位。
        self.mtp_step = mtp_step
        self._logical_to_storage_layer = {
            logical_layer_id: storage_layer_index
            for storage_layer_index, logical_layer_id in enumerate(config.logical_layer_ids)
        }
        from ..storage import allocate_buffers

        self.mtp_state_index = None
        self.bind_runtime_storage(allocate_buffers(self.get_runtime_storage_specs(), "meta"))
        self.big_pages_pool = None
        self.small_pages_pool = None

    def get_state_type(self) -> str:

        return (
            f"{self.algorithm}_conv_{dtype_name(dtype=self.config.conv_dtype)}"
            f"_ssm_{dtype_name(dtype=self.config.ssm_dtype)}"
        )

    def get_checkpoint_bytes(self):
        c = self.config
        return len(c.logical_layer_ids) * (
            c.conv_channels * c.conv_width * c.conv_dtype.itemsize
            + c.num_v_heads * c.key_dim * c.value_dim * c.ssm_dtype.itemsize
        )

    def get_runtime_storage_specs(self):
        from ..storage import TensorStorageSpec

        c, layers, requests = self.config, len(self.config.logical_layer_ids), self.request_capacity
        specs = (
            TensorStorageSpec(
                "conv_state", (layers, requests, c.conv_channels, c.conv_width + self.mtp_step), c.conv_dtype, fill=0
            ),
            TensorStorageSpec(
                "ssm_state",
                (layers, requests * (self.mtp_step + 1), c.num_v_heads, c.key_dim, c.value_dim),
                c.ssm_dtype,
                fill=0,
            ),
        )
        if self.mtp_step:
            specs += (TensorStorageSpec("mtp_state_index", (requests,), torch.int32, fill=0),)
        return specs

    def get_snapshot_storage_specs(self):
        from ..storage import TensorStorageSpec

        c, layers = self.config, len(self.config.logical_layer_ids)
        return (
            TensorStorageSpec("conv_state", (layers, c.conv_channels, c.conv_width), c.conv_dtype),
            TensorStorageSpec("ssm_state", (layers, c.num_v_heads, c.key_dim, c.value_dim), c.ssm_dtype),
        )

    # ========================================================================
    # 二、Bounded attention：计算视图、运行态重置、大小页与 MTP 提交
    # ========================================================================

    def contains_logical_layer(self, logical_layer_id: int) -> bool:
        """判断指定逻辑层是否属于当前 State 的层映射，共享层同样返回 True。"""
        return logical_layer_id in self._logical_to_storage_layer

    def get_layer_view(self, logical_layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """按逻辑层号映射到物理层，返回 Conv/SSM Tensor 引用；MTP 游标和步数直接从 State 读取。"""
        storage_layer_index = self._logical_to_storage_layer[logical_layer_id]
        return self.conv_state[storage_layer_index], self.ssm_state[storage_layer_index]

    # ========================================================================
    # 三、CPU Cache 数据交换
    # ========================================================================

    @deepcopy_result
    @weak_cached_method
    def get_cpu_transfer_regions(self) -> tuple[TransferRegion, ...]:
        """返回已接受 Conv/SSM 快照的字段区间与切片，供 CPU 页布局和容量计算使用。

        region.shape 描述 CPU 页中的全局字段；axis_mappings 将本地页池的 TP 分片映射到该字段。
        起止位置相对本 State，由管理器平移到 CPU 联合页；不包含 MTP 候选区或游标。
        """
        return self._get_transfer_regions()

    # ========================================================================
    # 五、PD 分离传输
    # ========================================================================

    @deepcopy_result
    @weak_cached_method
    def get_pd_transfer_regions(self) -> tuple[TransferRegion, ...]:
        """返回 PD 传输字段与切片，交由管理器安排其在联合页中的位置。"""
        return self._get_transfer_regions()

    # ========================================================================
    # 六、私有辅助实现
    # ========================================================================

    def _get_transfer_regions(self) -> tuple[TransferRegion, ...]:
        """依次描述全局 Conv 和 SSM 区域，并标明当前 TP rank 的读写范围。"""
        # 阶段一：确定全局 Conv 形状；只传有效历史窗口，不包含 MTP 候选区和游标。
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

        # 阶段三：SSM 沿 V head 切分；接在 Conv 之后，按 SSM dtype 对齐字节起点。
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
            self.state_name,
            head_start,
            config.num_v_heads,
            write_enabled,
            ssm_region.byte_start,
            ssm_region.byte_end,
        )
        return tuple(regions)

    def get_runtime_transfer_view(self):
        return BatchedStateView(
            (
                (
                    "conv",
                    TensorField(
                        self.conv_state.movedim(1, 0)[..., : self.config.conv_width],
                        ("slot", "layer", "channel", "window"),
                    ),
                ),
                (
                    "ssm",
                    TensorField(
                        self.ssm_state[:, :: self.mtp_step + 1].movedim(1, 0),
                        ("slot", "layer", "head", "key_dim", "value_dim"),
                    ),
                ),
            ),
            self.mtp_state_index,
            (
                ("conv", self.conv_state.element_size()),
                ("ssm", self.ssm_state.stride(1) * self.ssm_state.element_size()),
            ),
        )

    def get_snapshot_transfer_view(self, page_kind):
        pool = self.big_pages_pool if page_kind == "big" else self.small_pages_pool
        if pool is None:
            return BatchedStateView(())
        return BatchedStateView(
            (
                ("conv", TensorField(pool.conv_state, ("slot", "layer", "channel", "window"))),
                ("ssm", TensorField(pool.ssm_state, ("slot", "layer", "head", "key_dim", "value_dim"))),
            ),
            execution_device=self.conv_state.device,
        )
