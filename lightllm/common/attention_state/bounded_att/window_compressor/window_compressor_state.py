"""固定窗口、固定产出步长的 attention 压缩器候选/gate 运行态。"""

from __future__ import annotations

from lightllm.common.attention_state.utils import weak_cached_method

from dataclasses import dataclass
from math import prod
from typing import Literal

import torch

from lightllm.utils.envs_utils import get_env_start_args

from ...utils import deepcopy_result
from ...types import TensorField, TransferRegion
from ..base_state import BoundedAttentionState
from ..copy_plan import BatchedStateView
from ...storage import StatePagesPool


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
        assert self.target in ("kv", "indexer") and self.dtype in (
            torch.float32,
            torch.float16,
            torch.bfloat16,
        )
        assert 0 <= self.tp_rank < self.tp_world_size


class WindowCompressorState(BoundedAttentionState):
    """只保存窗口候选/gate，不保存压缩后的 KV 或 SWA。

    大小页在产出步长的边界保存重叠历史，PD 则保存完整窗口以支持任意前缀长度。
    无重叠的窗口在组边界不需要快照，因此不创建大小页池，也不占 CPU 联合页字节。
    """

    runtime_storage_fields = ("buffer",)

    @property
    def resource_kind(self):
        return "compressor_" + self.config.target

    # ========================================================================
    # 一、资源定义与初始化
    # ========================================================================

    def __init__(
        self,
        config: WindowCompressorStateConfig,
        request_capacity: int,
        mtp_step: int = 0,
    ) -> None:
        self.config = config
        self.state_name: str | None = None
        self.request_capacity = request_capacity
        self.mtp_step = mtp_step
        self._logical_to_storage_layer = {layer: index for index, layer in enumerate(config.logical_layer_ids)}
        from ...storage import allocate_buffers

        self.bind_runtime_storage(allocate_buffers(self.get_runtime_storage_specs(), "meta"))
        self.big_pages_pool: StatePagesPool | None = None
        self.small_pages_pool: StatePagesPool | None = None

    def get_state_type(self) -> str:
        """用途、窗口、步长和精度共同标识可复用的压缩器方案。"""
        precision = {torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16"}[self.config.dtype]
        return (
            f"compressor_{self.config.target}_w{self.config.window_size}" f"_s{self.config.compress_stride}_{precision}"
        )

    def get_checkpoint_bytes(self):
        c = self.config
        return len(c.logical_layer_ids) * c.overlap_tokens * 2 * c.candidate_dim * c.dtype.itemsize

    def get_runtime_storage_specs(self):
        from ...storage import TensorStorageSpec

        c = self.config
        shape = (len(c.logical_layer_ids), self.request_capacity, c.window_size + self.mtp_step, 2 * c.candidate_dim)
        hold_gate = (slice(None), self.request_capacity - 1, slice(None), slice(c.candidate_dim, None))
        return (TensorStorageSpec("buffer", shape, c.dtype, fill=0, fill_regions=((hold_gate, float("-inf")),)),)

    def get_snapshot_storage_specs(self):
        from ...storage import TensorStorageSpec

        c = self.config
        if not c.overlap_tokens:
            return ()
        return (
            TensorStorageSpec("buffer", (len(c.logical_layer_ids), c.overlap_tokens, 2 * c.candidate_dim), c.dtype),
        )

    # ========================================================================
    # 二、运行态：层视图、重置与大小页保存恢复
    # ========================================================================

    def contains_logical_layer(self, logical_layer_id: int) -> bool:
        return logical_layer_id in self._logical_to_storage_layer

    def get_layer_view(self, logical_layer_id: int) -> torch.Tensor:
        """返回该层 [request, window+MTP, candidate|gate] 的计算视图。"""
        return self.buffer[self._logical_to_storage_layer[logical_layer_id]]

    # ========================================================================
    # 三、CPU Cache：仅搬运时间有序的大小页快照
    # ========================================================================

    @deepcopy_result
    @weak_cached_method
    def get_cpu_transfer_regions(self) -> tuple[TransferRegion, ...]:
        """校验缓存页对齐产出步长；无重叠历史的窗口不占 CPU 页，也不创建大小页池。"""
        args = get_env_start_args()
        # 大页是小页的整数倍；小页对齐产出步长后，大小页终点均不会落在未完成的组内。
        assert (
            args.linear_att_hash_page_size % self.config.compress_stride == 0
        ), "state cache pages must align to the compressor stride"
        # CPU 服务只查询 meta 布局，也要在忽略无重叠窗口之前拒绝不安全的恢复边界。
        if args.cpu_cache_token_page_size is not None:
            assert (
                args.cpu_cache_token_page_size % self.config.compress_stride == 0
            ), "CPU cache pages must align to the compressor stride"
        return self._get_transfer_regions(runtime=False)

    # ========================================================================
    # 五、PD：按绝对前缀展开完整逻辑窗口，兼容不同 MTP 容量
    # ========================================================================

    @deepcopy_result
    @weak_cached_method
    def get_pd_transfer_regions(self) -> tuple[TransferRegion, ...]:
        """返回完整逻辑窗口，不包含本地 MTP 候选区。"""
        return self._get_transfer_regions(runtime=True)

    # ========================================================================
    # 六、内部页池和窗口搬运
    # ========================================================================

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

    def get_runtime_transfer_view(self):
        return BatchedStateView(
            (("compressor", TensorField(self.buffer.movedim(1, 0), ("slot", "layer", "window", "channel"))),)
        )

    def get_snapshot_transfer_view(self, page_kind):
        pool = self.big_pages_pool if page_kind == "big" else self.small_pages_pool
        return BatchedStateView(
            () if pool is None else (("compressor", TensorField(pool.buffer, ("slot", "layer", "window", "channel"))),),
            execution_device=self.buffer.device,
        )
