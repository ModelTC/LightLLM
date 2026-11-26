import math
from dataclasses import dataclass
from typing import Optional, Union

import torch


KV_QUANT_KINDS = (None, "fp8_per_tensor", "fp8_per_head", "int8_group", "int4_group")


@dataclass(frozen=True)
class GQAKVScales:
    """一个层的量化 KV 缓存的反量化比例因子（参见 GQAAttConfig.kv_quant）。

    - fp8_per_tensor: k_scale/v_scale 是主机浮点数。
    - fp8_per_head: k_scale/v_scale 是 float32 [num_kv_heads] 张量；q_scale 是
      [num_kv_heads] 静态查询比例因子，当层配置具有 has_q_scale 时，否则为 None。
    - int8_group/int4_group: k_scale/v_scale 是 [cache_slots, num_kv_heads,
      head_dim // kv_quant_group_size] 张量，地址方式与缓存相同。
    """

    k_scale: Union[float, torch.Tensor]
    v_scale: Union[float, torch.Tensor]
    q_scale: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class GQAAttConfig:
    """分组查询 attention（含 MHA 和 MQA）的静态层配置。

    注意力头数按 rank 计算，num_heads 必须是 num_kv_heads 的正整数倍。
    softmax_scale 默认为 1/sqrt(head_dim)。

    causal=True 时，每个 query 只看自身位置及其之前的 key。causal=False
    允许同一请求的 query 看到它的全部 key，满足块草稿模型的需要。
    sliding_window=(-1, -1) 时使用全部可见 key。(left, right) 的两侧都 >= 0 时，
    只使用 [query_pos - left, query_pos + right] 内的 key。因果 attention 使用
    (left, 0)，只有非因果 attention（块草稿）可以向右看。任意两个元素的序列都会转成元组，
    所以从 JSON 模型配置读出的列表与对应的元组相等。

    has_sinks=True 时，每个 query 头的 softmax 都加入一个学习得到的 sink logit，
    由每层的 GQAAttInputs.sinks 提供。

    kv_quant 指定 K/V 的缓存格式，None 表示按 dtype 存储。
    "fp8_per_tensor" 和 "fp8_per_head" 使用 FP8 E4M3，每层或每个 KV 头有一个静态 scale。
    "int8_group" 和 "int4_group" 使用整数格式，每个 token、KV 头和 kv_quant_group_size
    个通道有一个动态 scale。各层通过 GQAAttInputs.kv_scales 传入 scale。
    has_q_scale 用于 "fp8_per_head"。True 使用每个 KV 头的静态 scale 量化 q，
    False 按请求动态量化。

    has_alibi=True 时加入 ALiBi 位置偏置，每个 query 头有一个斜率，
    由每层的 GQAAttInputs.alibi_slopes 提供。

    prefill 时，bidirectional_image_spans=True 允许图像区间里的 query 看到同一区间内
    的全部 key。InferStateInfo.b_image_token_end 给出本步区间的终点。滑动窗口还需要
    b_image_token_start，窗口之外的同区间 key 仍可见。文本 query 继续使用因果约束。

    各实现的说明列出了支持的 dtype、头维度、scale 和滑动窗口模式。
    """

    num_heads: int
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype
    sliding_window: tuple[int, int] = (-1, -1)
    softmax_scale: Optional[float] = None
    causal: bool = True
    has_sinks: bool = False
    kv_quant: Optional[str] = None
    kv_quant_group_size: Optional[int] = None
    has_q_scale: bool = False
    has_alibi: bool = False
    bidirectional_image_spans: bool = False

    def __post_init__(self) -> None:
        if self.num_heads <= 0 or self.num_kv_heads <= 0 or self.head_dim <= 0:
            raise ValueError("num_heads, num_kv_heads and head_dim must be positive")
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be a positive multiple of num_kv_heads ({self.num_kv_heads})"
            )
        left, right = (int(x) for x in self.sliding_window)
        if (left, right) != (-1, -1) and (left < 0 or right < 0):
            raise ValueError(f"sliding_window must be (-1, -1) or have both sides >= 0, got {(left, right)}")
        if self.causal and right > 0:
            raise ValueError(f"a causal sliding_window cannot look right, got {(left, right)}")
        object.__setattr__(self, "sliding_window", (left, right))
        if self.softmax_scale is None:
            object.__setattr__(self, "softmax_scale", 1.0 / math.sqrt(self.head_dim))
        if self.kv_quant not in KV_QUANT_KINDS:
            raise ValueError(f"kv_quant must be one of {KV_QUANT_KINDS}, got {self.kv_quant!r}")
        grouped = self.kv_quant in ("int8_group", "int4_group")
        if grouped != (self.kv_quant_group_size is not None):
            raise ValueError("kv_quant_group_size must be given exactly for the int8/int4 group kinds")
        if grouped and (self.kv_quant_group_size <= 0 or self.head_dim % self.kv_quant_group_size):
            raise ValueError(f"kv_quant_group_size must divide head_dim, got {self.kv_quant_group_size}")
        if self.has_q_scale and self.kv_quant != "fp8_per_head":
            raise ValueError("has_q_scale applies only to kv_quant='fp8_per_head'")

    @property
    def has_default_scale(self) -> bool:
        return math.isclose(self.softmax_scale, 1.0 / math.sqrt(self.head_dim), rel_tol=1e-6)


@dataclass(frozen=True)
class GQAAttInputs:
    """分组查询 attention 使用的逐层输入，由调用方持有。

    prefill 时 q 的形状为 [total_query_tokens, num_heads, head_dim]，按批内顺序打包。
    decode 时形状为 [decode_rows, num_heads, head_dim]，与批内行逐一对应，
    每请求的最大查询行数由后端的运行时配置指定。

    k_cache/v_cache 是已填充的逐层物理缓存视图，形状为 [cache_slots, num_kv_heads, head_dim]。
    att_config 的 kv_quant 决定编码方式，可使用 dtype、FP8 E4M3（或其 uint8 字节表示）、
    int8，或每字节打包两个 int4 值的 int8（[..., head_dim // 2]）。
    各实现的 attention 状态索引其第 0 维。缓存由调用方持有并填入数据，包括当前 query
    的 K/V。后端读取 q 和 K/V。

    att_config 的 has_sinks 启用时，sinks 是 dtype 与 q 相同的 [num_heads] sink logits，
    否则为 None。配置 kv_quant 时，kv_scales 持有缓存的 scale，否则为 None。
    配置 has_alibi 时，alibi_slopes 是本层的 [num_heads] float32 ALiBi 斜率，否则为 None。
    """

    q: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    sinks: Optional[torch.Tensor] = None
    kv_scales: Optional[GQAKVScales] = None
    alibi_slopes: Optional[torch.Tensor] = None


_GQA_KV_QUANT = {
    "None": None,
    "int8kv": "int8_group",
    "int4kv": "int4_group",
    "fp8kv_sph": "fp8_per_head",
    "fp8kv_spt": "fp8_per_tensor",
}


def get_gqa_kv_quant(
    llm_kv_type: str, kv_scales: Optional[GQAKVScales], head_dim: int
) -> tuple[str | None, int | None, bool]:
    """返回量化方式、每组宽度，以及是否使用静态 Q scale。"""
    if llm_kv_type not in _GQA_KV_QUANT:
        raise ValueError(f"--llm_kv_type {llm_kv_type!r} has no GQA KV cache layout; choose from {list(_GQA_KV_QUANT)}")
    kv_quant = _GQA_KV_QUANT[llm_kv_type]
    if (kv_quant is None) != (kv_scales is None):
        raise ValueError(f"--llm_kv_type {llm_kv_type!r} does not match the cache's scales")
    grouped = kv_quant in ("int8_group", "int4_group")
    group_size = head_dim // kv_scales.k_scale.shape[-1] if grouped else None
    has_q_scale = kv_quant == "fp8_per_head" and kv_scales.q_scale is not None
    return kv_quant, group_size, has_q_scale
