from dataclasses import dataclass

import torch

from lightllm.common.attention.backend import AttentionBackend, TensorAllocator
from . import GQAInputs


@dataclass(frozen=True)
class TritonGQAPrefillBatch:
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
class TritonGQADecodeBatch:
    """Metadata for one query per request, in b_req_idx order.

    Index tensors follow TritonGQAPrefillBatch's layout/device contract.
    b_seq_len gives actual positive lengths, including the current token;
    all addressed token-map entries must identify populated cache slots.

    kv_seq_len_upper_bound bounds actual lengths for MHA scratch sizing and
    launch coverage; a batch planned for CUDA graph capture must use the graph
    capacity. tuning_kv_seq_len is only a GQA configuration hint and need not
    equal the current maximum during replay: a hint of 33 and capacity of 512
    remain valid when actual lengths later reach 300.
    """

    req_to_token_indexs: torch.Tensor
    b_req_idx: torch.Tensor
    b_seq_len: torch.Tensor
    kv_seq_len_upper_bound: int
    tuning_kv_seq_len: int


class TritonGQAPrefillBackend(AttentionBackend[TritonGQAPrefillBatch, GQAInputs]):
    def plan(
        self,
        *,
        req_to_token_indexs: torch.Tensor,
        b_req_idx: torch.Tensor,
        b_seq_len: torch.Tensor,
        b_q_start_loc: torch.Tensor,
        b_ready_cache_len: torch.Tensor,
        max_q_seq_len: int,
    ) -> TritonGQAPrefillBatch:
        """Kernels read the metadata tensors directly, so planning only records them."""
        return TritonGQAPrefillBatch(
            req_to_token_indexs=req_to_token_indexs,
            b_req_idx=b_req_idx,
            b_seq_len=b_seq_len,
            b_q_start_loc=b_q_start_loc,
            b_ready_cache_len=b_ready_cache_len,
            max_q_seq_len=max_q_seq_len,
        )

    def forward(
        self,
        batch: TritonGQAPrefillBatch,
        inputs: GQAInputs,
        *,
        out: torch.Tensor,
        alloc_func: TensorAllocator = torch.empty,
    ) -> torch.Tensor:
        """Fill and return out, with q's shape, dtype and device.

        q, K/V and out are matching FP16/BF16 tensors on one CUDA device, with
        unit head-dimension stride and non-overlapping token/head strides that
        may include padding. K/V may be disjoint slices of shared storage;
        arbitrary strided layouts are not supported. The caller must make q.device
        the current CUDA device. out must not alias inputs or batch tensors.
        Head dimensions 16, 32, 64, 128, 256 and 512 are supported.
        sliding_window (-1, -1) and (left, 0) are supported.
        Prefill needs no scratch and does not call alloc_func.
        """
        from lightllm.common.attention.kernels.gqa.bf16.prefill.context_flashattention_nopad import (
            context_attention_fwd,
        )

        context_attention_fwd(
            q=inputs.q,
            k=inputs.k_cache,
            v=inputs.v_cache,
            o=out,
            b_req_idx=batch.b_req_idx,
            b_start_loc=batch.b_q_start_loc,
            b_seq_len=batch.b_seq_len,
            b_prompt_cache_len=batch.b_ready_cache_len,
            max_input_len=batch.max_q_seq_len,
            req_to_token_indexs=batch.req_to_token_indexs,
            sliding_window=inputs.sliding_window,
        )
        return out


class TritonGQADecodeBackend(AttentionBackend[TritonGQADecodeBatch, GQAInputs]):
    def plan(
        self,
        *,
        req_to_token_indexs: torch.Tensor,
        b_req_idx: torch.Tensor,
        b_seq_len: torch.Tensor,
        kv_seq_len_upper_bound: int,
        tuning_kv_seq_len: int,
    ) -> TritonGQADecodeBatch:
        """Kernels read the metadata tensors directly, so planning only records them and replay needs no refresh."""
        return TritonGQADecodeBatch(
            req_to_token_indexs=req_to_token_indexs,
            b_req_idx=b_req_idx,
            b_seq_len=b_seq_len,
            kv_seq_len_upper_bound=kv_seq_len_upper_bound,
            tuning_kv_seq_len=tuning_kv_seq_len,
        )

    def forward(
        self,
        batch: TritonGQADecodeBatch,
        inputs: GQAInputs,
        *,
        out: torch.Tensor,
        alloc_func: TensorAllocator = torch.empty,
    ) -> torch.Tensor:
        """Fill and return out, with q's shape, dtype and device.

        q, K/V and out are matching FP16/BF16 tensors on one CUDA device, with
        unit head-dimension stride and non-overlapping token/head strides that
        may include padding. K/V may be disjoint slices of shared storage;
        arbitrary strided layouts are not supported. The caller must make q.device
        the current CUDA device. out must not alias inputs or batch tensors.
        K and V must have identical strides.
        Head dimensions 16, 32, 64, 128 and 256 are supported, plus 512 for GQA.
        GQA (q_heads > kv_heads) supports sliding_window (-1, -1) and (left, 0);
        MHA (q_heads == kv_heads) supports only (-1, -1).
        Scratch uses alloc_func on q.device. A serving allocator with a cache
        scoped to one device must be used only on that device.
        """
        q_head_num = inputs.q.shape[1]
        k_head_num = inputs.k_cache.shape[1]
        if q_head_num % k_head_num != 0:
            raise ValueError(f"q_heads ({q_head_num}) must be a positive multiple of kv_heads ({k_head_num})")
        if q_head_num == k_head_num:
            assert inputs.sliding_window == (-1, -1), "sliding_window not supported in non-gqa attention yet"
            from lightllm.common.attention.kernels.gqa.bf16.decode.flash_decoding import (
                token_decode_attention_flash_decoding,
            )

            return token_decode_attention_flash_decoding(
                q=inputs.q,
                cache_k=inputs.k_cache,
                cache_v=inputs.v_cache,
                req_to_token_indexs=batch.req_to_token_indexs,
                b_req_idx=batch.b_req_idx,
                b_seq_len=batch.b_seq_len,
                max_kv_seq_len=batch.kv_seq_len_upper_bound,
                out=out,
                alloc_tensor_func=alloc_func,
            )
        else:
            from lightllm.common.attention.kernels.gqa.bf16.decode.gqa_flash_decoding import (
                gqa_token_decode_attention_flash_decoding,
            )

            return gqa_token_decode_attention_flash_decoding(
                q=inputs.q,
                cache_k=inputs.k_cache,
                cache_v=inputs.v_cache,
                req_to_token_indexs=batch.req_to_token_indexs,
                b_req_idx=batch.b_req_idx,
                b_seq_len=batch.b_seq_len,
                max_len_in_batch=batch.tuning_kv_seq_len,
                out=out,
                alloc_tensor_func=alloc_func,
                sliding_window=inputs.sliding_window,
            )
