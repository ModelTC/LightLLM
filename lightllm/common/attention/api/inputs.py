"""区分初始化时绑定的层资源与每次 forward 的 tensor 输入。

*Params 随 LayerAttentionBinding 保存，借用权重以及 State 管理的 KV / Conv / SSM 视图；
*Inputs 由模型层为当前调用构造，批次级请求索引和 provider 派生 metadata 则由 plan 提供。
这些数据结构只固定字段绑定，不禁止 backend 原地更新所借用的状态 tensor。
"""

from dataclasses import dataclass
from typing import NamedTuple, Optional

import torch


@dataclass(frozen=True)
class DenseParams:
    key: torch.Tensor
    value: torch.Tensor
    alibi: Optional[torch.Tensor] = None
    sliding_window: Optional[tuple[int, int]] = None
    sink: Optional[torch.Tensor] = None
    query_scale: Optional[torch.Tensor] = None
    key_scale: torch.Tensor | float | None = None
    value_scale: torch.Tensor | float | None = None


@dataclass(frozen=True)
class MLAParams:
    kv: torch.Tensor
    softmax_scale: float
    kv_lora_rank: int
    qk_rope_head_dim: int


@dataclass(frozen=True)
class GDNParams:
    conv: torch.Tensor
    ssm: torch.Tensor
    conv_weight: torch.Tensor
    conv_bias: Optional[torch.Tensor]
    a_log: torch.Tensor
    dt_bias: torch.Tensor


@dataclass(frozen=True)
class KDAParams:
    conv: torch.Tensor
    ssm: torch.Tensor
    conv_weight: torch.Tensor
    a_log: torch.Tensor
    dt_bias: torch.Tensor


@dataclass(frozen=True)
class DenseInputs:
    """Dense 路径在调用前已写入 KV cache，因此这里仅传 query。"""

    query: torch.Tensor


class VarlenAttentionInputs(NamedTuple):
    """非因果 self-attention 的 Q/K/V，布局均为 [total_tokens, heads, head_dim]。

    cu_seqlens 是同设备上的 int32 前缀和，始于 0、终于 total_tokens；相邻边界之间
    独立做 attention，允许空序列。max_seqlen 是最长序列长度，由调用方提供以避免 GPU 同步。
    每层只打包 tensor 引用，使用 NamedTuple 保持字段不可变并减少构造开销。
    """

    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    cu_seqlens: torch.Tensor
    max_seqlen: int


@dataclass(frozen=True)
class MLAPrefillInputs:
    query: torch.Tensor
    key_nope: torch.Tensor
    key_rope: torch.Tensor
    value: torch.Tensor


@dataclass(frozen=True)
class MLADecodeInputs:
    query_nope: torch.Tensor
    query_rope: torch.Tensor


@dataclass(frozen=True)
class SparseMLAPrefillInputs:
    query: torch.Tensor
    topk_mem_indices: torch.Tensor
    topk_indices: Optional[torch.Tensor] = None
    prefill_cache_kv: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class SparseMLADecodeInputs:
    query_nope: torch.Tensor
    query_rope: torch.Tensor
    topk_mem_indices: torch.Tensor


@dataclass(frozen=True)
class GDNInputs:
    mixed_qkvzba: torch.Tensor


@dataclass(frozen=True)
class KDAInputs:
    mixed_qkv: torch.Tensor
    raw_gate: torch.Tensor
    raw_beta: torch.Tensor
