from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MLAPrefillAttConfig:
    """使用展开 K/V（未吸收）的 MLA prefill 静态配置。

    注意力头数按 rank 计算。query 和 key 有 qk_nope_head_dim + qk_rope_head_dim 个通道，
    value 和输出有 v_head_dim 个通道。prefill 和 decode 对同一层使用不同投影，因此层配置
    各自定义。causal=False 允许同一请求的 query 看到其所有 key，用于块草稿模型。
    """

    num_heads: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    softmax_scale: float
    dtype: torch.dtype
    causal: bool = True

    def __post_init__(self) -> None:
        if self.num_heads <= 0 or self.qk_nope_head_dim <= 0 or self.qk_rope_head_dim <= 0 or self.v_head_dim <= 0:
            raise ValueError("num_heads, qk_nope_head_dim, qk_rope_head_dim and v_head_dim must be positive")


@dataclass(frozen=True)
class MLAPrefillAttInputs:
    """按请求顺序排列的每层投影键值，包括已缓存前缀。

    q 是 [query_tokens, num_heads, qk_nope_head_dim + qk_rope_head_dim]。k_nope 是
    [kv_tokens, num_heads, qk_nope_head_dim]，k_rope 是 [kv_tokens, 1, qk_rope_head_dim]，v 是 [kv_tokens,
    num_heads, v_head_dim]。
    """

    q: torch.Tensor
    k_nope: torch.Tensor
    k_rope: torch.Tensor
    v: torch.Tensor


@dataclass(frozen=True)
class MLADecodeAttConfig:
    """读取压缩 latent cache 的 MLA decode 静态配置。

    注意力头数按 rank 计算。吸收投影后的 query 有 kv_lora_rank 个 latent 通道和
    qk_rope_head_dim 个 RoPE 通道，与单行缓存的布局一致。投机解码产生多行 query 时，
    因果约束由该配置控制，每请求的最大查询行数由后端的运行时配置指定。
    """

    num_heads: int
    kv_lora_rank: int
    qk_rope_head_dim: int
    softmax_scale: float
    dtype: torch.dtype
    causal: bool = True

    def __post_init__(self) -> None:
        if self.num_heads <= 0 or self.kv_lora_rank <= 0 or self.qk_rope_head_dim <= 0:
            raise ValueError("num_heads, kv_lora_rank and qk_rope_head_dim must be positive")


@dataclass(frozen=True)
class MLADecodeAttInputs:
    """吸收投影后的 query 行，分为 latent 和 RoPE 两部分。

    q_nope 的形状为 [rows, num_heads, kv_lora_rank]，q_rope 的形状为
    [rows, num_heads, qk_rope_head_dim]。kv_cache 是调用方持有的缓存，形状为
    [physical_slots, 1, kv_lora_rank + qk_rope_head_dim]。其中的 latent 前缀同时提供 K 和 V。
    缓存包含各请求的当前 token。
    """

    q_nope: torch.Tensor
    q_rope: torch.Tensor
    kv_cache: torch.Tensor
