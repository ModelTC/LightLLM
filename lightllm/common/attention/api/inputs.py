"""区分初始化时绑定的层资源与每次 forward 的 tensor 输入。

*Params 随 LayerAttentionBinding 保存，借用权重以及 State 管理的 KV / Conv / SSM 视图；
*Inputs 由模型层为当前调用构造，批次级请求索引和 provider 派生 metadata 则由 context 提供。
这些数据结构只固定字段绑定，不禁止 backend 原地更新所借用的状态 tensor。
"""

from dataclasses import dataclass
from typing import NamedTuple, Optional, Protocol

import torch


@dataclass(frozen=True)
class DenseParams:
    key: torch.Tensor
    value: torch.Tensor
    alibi: Optional[torch.Tensor] = None
    sink: Optional[torch.Tensor] = None
    query_scale: Optional[torch.Tensor] = None
    key_scale: torch.Tensor | float | None = None
    value_scale: torch.Tensor | float | None = None


@dataclass(frozen=True)
class MLAParams:
    kv: torch.Tensor
    kv_lora_rank: int
    qk_rope_head_dim: int


class ReplaySSMState(Protocol):
    """Borrowed replay storage; State retains allocation, checkpoint and release ownership."""

    ssm_state: torch.Tensor
    keys: torch.Tensor
    deltas: torch.Tensor
    gates: torch.Tensor
    raw_keys: torch.Tensor
    raw_values: torch.Tensor
    betas: torch.Tensor
    cursors: torch.Tensor
    tile_sync_mark: torch.Tensor
    num_key_heads: int
    capacity: int
    verify_width: int
    compute_mode: str
    gate_lower_bound: float
    hold_req_id: int
    kda: bool

    def storage_layer(self, logical_layer_id: int) -> int:
        ...

    def finish_decode_token(self, logical_layer_id: int, reqs: torch.Tensor, cu_seqlens: torch.Tensor) -> None:
        ...


@dataclass(frozen=True)
class GDNParams:
    conv: torch.Tensor
    ssm: torch.Tensor
    conv_weight: torch.Tensor
    conv_bias: Optional[torch.Tensor]
    a_log: torch.Tensor
    dt_bias: torch.Tensor
    mtp_state_index: torch.Tensor | None = None
    replay_state: ReplaySSMState | None = None


@dataclass(frozen=True)
class KDAParams:
    conv: torch.Tensor
    ssm: torch.Tensor
    conv_weight: torch.Tensor
    a_log: torch.Tensor
    dt_bias: torch.Tensor
    mtp_state_index: torch.Tensor | None = None
    replay_state: ReplaySSMState | None = None


@dataclass(frozen=True)
class DenseInputs:
    """Dense 路径在调用前已写入 KV cache，因此这里仅传 query。"""

    query: torch.Tensor


class VarlenAttentionInputs(NamedTuple):
    """非因果 self-attention 的 Q/K/V，布局均为 [total_tokens, heads, head_dim]。

    cu_seqlens 是同设备上的 int32 前缀和，始于 0、终于 total_tokens；相邻边界之间
    独立做 attention，允许空序列。max_seqlen 是最长序列长度的容量上界，由调用方提供以避免 GPU 同步。
    每层只打包 tensor 引用，使用 NamedTuple 保持字段不可变并减少构造开销。
    """

    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    cu_seqlens: torch.Tensor
    max_seqlen: int

    def validate(self) -> None:
        """Check sequence boundaries without copying CUDA lengths to the host.

        max_seqlen is an upper bound (Graph buckets may reserve a larger value).
        CUDA value checks are asynchronous device assertions, including in Graphs;
        invalid boundaries are input errors, not a reason to try another provider.
        """
        q, k, v, cu, maximum = self
        if q.ndim != 3 or q.shape != k.shape or q.shape != v.shape:
            raise ValueError("varlen self-attention requires matching [tokens, heads, dim] Q/K/V")
        if q.device != k.device or q.device != v.device or q.dtype != k.dtype or q.dtype != v.dtype:
            raise ValueError("Q/K/V must have the same dtype and device")
        if cu.ndim != 1 or cu.numel() < 2 or cu.dtype != torch.int32 or cu.device != q.device:
            raise ValueError("cu_seqlens must be a same-device int32 vector with at least two entries")
        if maximum < 0 or (q.shape[0] and maximum == 0):
            raise ValueError("nonempty attention requires a positive max_seqlen")
        lengths = cu[1:] - cu[:-1]
        valid = (cu[0] == 0) & (cu[-1] == q.shape[0]) & (lengths >= 0).all() & (lengths <= maximum).all()
        message = "cu_seqlens must span all tokens with nonnegative lengths bounded by max_seqlen"
        if cu.device.type == "cuda":
            torch._assert_async(valid, message)
        elif not bool(valid):
            raise ValueError(message)


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


class GDNOutput(NamedTuple):
    hidden_states: torch.Tensor
    gate: torch.Tensor
