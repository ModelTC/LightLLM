"""模型选择 V4 的逻辑层和压缩方案，State 配置只记录确定的几何参数。"""


from dataclasses import dataclass
from typing import ClassVar

from lightllm.common.attention_state.bounded_att.fp8_e8m0_swa_state import FP8E8M0SlidingWindowConfig
from lightllm.common.attention_state.full_att.compressed_att_state import CompressedAttentionConfig


@dataclass(frozen=True)
class DSV4AttentionConfig:
    # 当前模型拥有的逻辑层 ID，顺序对应 SWA 的物理 layer 轴；main/draft 分别创建。
    logical_layer_ids: tuple[int, ...]
    # 与 logical_layer_ids 逐项对应；0 为纯 SWA，4/128 为同时启用对应压缩器。
    compress_ratios: tuple[int, ...]
    # 未编码的共享 KV 向量宽度，包含以 BF16 存储的 RoPE 部分。
    head_dim: int
    # RoPE 特征宽度；其余特征按每 64 个元素一组编码为 FP8。
    rope_head_dim: int
    # C4 indexer key 的特征宽度；每个 key 另有一个 FP32 scale。
    indexer_head_dim: int
    # 每份可恢复 SWA 快照包含的逻辑 token 数，不包含 MTP 候选位置。
    sliding_window: int
    # 存储在各 TP rank 完整复制，只有 rank 0 向共享的 CPU/PD 区域写出。
    tp_rank: int = 0
    tp_world_size: int = 1

    # 压缩 KV 按原始 token 的 256-token 块存储，KV 与 indexer 不得跨块拆开。
    compressed_kv_token_alignment: ClassVar[int] = 256

    def __post_init__(self) -> None:
        assert self.logical_layer_ids and len(set(self.logical_layer_ids)) == len(self.logical_layer_ids)
        assert len(self.logical_layer_ids) == len(self.compress_ratios)
        assert all(ratio in (0, 4, 128) for ratio in self.compress_ratios)
        assert self.head_dim > self.rope_head_dim > 0 and (self.head_dim - self.rope_head_dim) % 64 == 0
        assert self.indexer_head_dim > 0 and self.sliding_window > 0
        assert 0 <= self.tp_rank < self.tp_world_size

    def create_swa_config(self) -> FP8E8M0SlidingWindowConfig:
        """模型选择窗口及向量维度，具体编码和物理页布局由原子配置负责。"""
        return FP8E8M0SlidingWindowConfig(
            logical_layer_ids=self.logical_layer_ids,
            window_size=self.sliding_window,
            kv_dim=self.head_dim,
            rope_dim=self.rope_head_dim,
            tp_rank=self.tp_rank,
            tp_world_size=self.tp_world_size,
        )

    def create_compressed_config(self, compress_ratio: int) -> CompressedAttentionConfig:
        """按模型层组合创建一种压缩历史配置，供 State 创建和 backend 使用。"""
        return CompressedAttentionConfig(
            logical_layer_groups=tuple(
                (layer,)
                for layer, ratio in zip(self.logical_layer_ids, self.compress_ratios)
                if ratio == compress_ratio
            ),
            compress_ratio=compress_ratio,
            kv_dim=self.head_dim,
            rope_dim=self.rope_head_dim,
            block_token_size=self.compressed_kv_token_alignment,
            indexer_dim=self.indexer_head_dim if compress_ratio == 4 else 0,
            tp_rank=self.tp_rank,
            tp_world_size=self.tp_world_size,
        )


def create_attention_config(
    model_config: dict[str, object],
    model_logical_layer_ids: tuple[int, ...],
    tp_rank: int = 0,
    tp_world_size: int = 1,
) -> DSV4AttentionConfig:
    ratios = model_config["compress_ratios"]
    return DSV4AttentionConfig(
        logical_layer_ids=model_logical_layer_ids,
        compress_ratios=tuple(ratios[layer] if layer < len(ratios) else 0 for layer in model_logical_layer_ids),
        head_dim=model_config["head_dim"],
        rope_head_dim=model_config["qk_rope_head_dim"],
        indexer_head_dim=model_config["index_head_dim"],
        sliding_window=model_config["sliding_window"],
        tp_rank=tp_rank,
        tp_world_size=tp_world_size,
    )
