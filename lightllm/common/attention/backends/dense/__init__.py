from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DenseInputs:
    """Borrowed operands for causal dense attention, with scale 1/sqrt(head_dim).

    q is [total_query_tokens, q_heads, head_dim] in prefill, packed in batch
    order, or [batch_size, q_heads, head_dim] in decode (one final-position
    query per request). q_heads must be a positive multiple of kv_heads.

    k_cache/v_cache are already populated per-layer physical cache views of
    shape [cache_slots, kv_heads, head_dim]. The batch token map addresses
    their first dimension. The caller owns and populates the cache, including
    the current queries' KV; these backends do not write q, K/V or metadata.

    Triton supports matching FP16/BF16 operands on one CUDA device, with unit
    head-dimension stride and non-overlapping token/head strides that may
    include padding. Decode requires identical K/V strides. K/V may be
    disjoint slices of shared storage; arbitrary strided layouts are not
    part of this contract. Select q.device as the current device for execution.

    sliding_window=(-1, -1) attends to the full causal prefix. (left, 0), with
    left >= 0, includes positions [max(0, query_pos-left), query_pos]. Prefill
    and GQA decode support this window; MHA decode supports only (-1, -1).
    """

    q: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    sliding_window: tuple[int, int] = (-1, -1)


@dataclass(frozen=True)
class DensePrefillBatch:
    """Metadata for packed queries after each request's cached prefix.

    req_to_token_indexs[request_id, position] is a physical KV cache slot,
    not a byte offset. It is a 2-D int32/int64 tensor with unit sequence stride.
    b_req_idx selects its rows; all b_* tensors are contiguous int32/int64
    vectors of batch_size elements on q.device, as is the token map.

    Request i has b_seq_len[i] total tokens including b_ready_cache_len[i]
    cached tokens. Its remaining queries occupy the packed q/out range
    starting at b_q_start_loc[i]. These ranges must be disjoint and cover q.
    max_q_seq_len bounds every b_seq_len[i] - b_ready_cache_len[i] for launch
    coverage; the caller supplies it without reading GPU lengths in forward.
    All addressed token-map entries must identify populated cache slots.
    """

    req_to_token_indexs: torch.Tensor
    b_req_idx: torch.Tensor
    b_seq_len: torch.Tensor
    b_q_start_loc: torch.Tensor
    b_ready_cache_len: torch.Tensor
    max_q_seq_len: int


@dataclass(frozen=True)
class DenseDecodeBatch:
    """Metadata for one query per request, in b_req_idx order.

    Index tensors follow DensePrefillBatch's layout/device contract.
    b_seq_len gives actual positive lengths, including the current token;
    all addressed token-map entries must identify populated cache slots.

    kv_seq_len_upper_bound bounds actual lengths for MHA scratch sizing and
    launch coverage. It may be CUDA graph capacity. tuning_kv_seq_len is only
    a GQA configuration hint, saved before capture rewrites the upper bound.
    It need not equal the current maximum during replay: a hint of 33 and
    capacity of 512 remain valid when actual lengths later reach 300.
    """

    req_to_token_indexs: torch.Tensor
    b_req_idx: torch.Tensor
    b_seq_len: torch.Tensor
    kv_seq_len_upper_bound: int
    tuning_kv_seq_len: int
