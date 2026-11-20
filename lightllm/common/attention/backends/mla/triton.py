from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from lightllm.common.attention.backend import AttentionBackend, TensorAllocator

if TYPE_CHECKING:
    from lightllm.common.attention.kernels.mla.decode.gqa_flash_decoding import MLADecodeSchedule


@dataclass(frozen=True)
class TritonMLAPrefillBatch:
    """Packed query and KV row offsets, total lengths and cached prefix lengths per request."""

    b_q_start_loc: torch.Tensor
    b_kv_start_loc: torch.Tensor
    b_seq_len: torch.Tensor
    b_ready_cache_len: torch.Tensor
    max_q_seq_len: int


@dataclass(frozen=True)
class MLAPrefillInputs:
    """Per-layer projected keys/values in request order, including cached prefixes.

    q is [query_tokens, heads, nope_dim + 64]. k_nope and v are
    [kv_tokens, heads, nope_dim]; k_rope is [kv_tokens, 1, 64].
    """

    q: torch.Tensor
    k_nope: torch.Tensor
    k_rope: torch.Tensor
    v: torch.Tensor
    softmax_scale: float


class TritonMLAPrefillBackend(AttentionBackend[TritonMLAPrefillBatch, MLAPrefillInputs]):
    def forward(
        self,
        batch: TritonMLAPrefillBatch,
        inputs: MLAPrefillInputs,
        *,
        out: torch.Tensor,
        alloc_func: TensorAllocator = torch.empty,
    ) -> torch.Tensor:
        """Write causal attention to out [query_tokens, heads, nope_dim] and return it.

        Operands share dtype/device and have unit last stride. out must be disjoint from inputs.
        """
        from lightllm.common.attention.kernels.mla.prefill.context_flashattention_nopad_with_v import (
            context_attention_fwd_with_v,
        )

        context_attention_fwd_with_v(
            q_nope=inputs.q[:, :, :-64],
            q_rope=inputs.q[:, :, -64:],
            k_nope=inputs.k_nope,
            k_rope=inputs.k_rope,
            v=inputs.v,
            o=out,
            b_start_loc=batch.b_q_start_loc,
            b_kv_start_loc=batch.b_kv_start_loc,
            b_seq_len=batch.b_seq_len,
            b_prompt_cache_len=batch.b_ready_cache_len,
            max_input_len=batch.max_q_seq_len,
            softmax_scale=inputs.softmax_scale,
        )
        return out


@dataclass(frozen=True)
class MLADecodeInputs:
    """One absorbed query per request, split into latent and rotary components.

    kv_cache is a borrowed [physical_slots, 1, latent_dim + 64] cache. Its latent
    prefix supplies both K and V. The cache includes the current token for each request.
    """

    q_nope: torch.Tensor
    q_rope: torch.Tensor
    kv_cache: torch.Tensor
    softmax_scale: float


@dataclass(frozen=True)
class TritonMLADecodeBatch:
    """At decode, request i reads physical slots req_to_token_indexs[b_req_idx[i], :b_seq_len[i]]."""

    req_to_token_indexs: torch.Tensor
    b_req_idx: torch.Tensor
    b_seq_len: torch.Tensor
    schedule: "MLADecodeSchedule"

    @classmethod
    def prepare(
        cls,
        inputs: MLADecodeInputs,
        req_to_token_indexs: torch.Tensor,
        b_req_idx: torch.Tensor,
        b_seq_len: torch.Tensor,
        tuning_seq_len: int,
    ) -> "TritonMLADecodeBatch":
        """Prepare the split schedule once for layers with the same kernel geometry.

        Run during graph capture so replay recomputes the splits from live b_seq_len.
        tuning_seq_len is a host-side configuration hint, independent of those lengths.
        """
        from lightllm.common.attention.kernels.mla.decode.gqa_flash_decoding import prepare_mla_decode_schedule

        schedule = prepare_mla_decode_schedule(
            q_nope=inputs.q_nope,
            q_rope=inputs.q_rope,
            kv_nope=inputs.kv_cache[:, :, :-64],
            kv_rope=inputs.kv_cache[:, :, -64:],
            req_to_token_indexs=req_to_token_indexs,
            b_req_idx=b_req_idx,
            b_seq_len=b_seq_len,
            tuning_seq_len=tuning_seq_len,
            softmax_scale=inputs.softmax_scale,
        )
        return cls(req_to_token_indexs=req_to_token_indexs, b_req_idx=b_req_idx, b_seq_len=b_seq_len, schedule=schedule)


class TritonMLADecodeBackend(AttentionBackend[TritonMLADecodeBatch, MLADecodeInputs]):
    def forward(
        self,
        batch: TritonMLADecodeBatch,
        inputs: MLADecodeInputs,
        *,
        out: torch.Tensor,
        alloc_func: TensorAllocator = torch.empty,
    ) -> torch.Tensor:
        """Fill and return out with q_nope's [requests, heads, latent_dim] shape, dtype and device.

        Inputs and cache are read-only. out has unit last stride and must not overlap them.
        """
        from lightllm.common.attention.kernels.mla.decode.gqa_flash_decoding import (
            gqa_token_decode_attention_flash_decoding,
        )

        gqa_token_decode_attention_flash_decoding(
            q_nope=inputs.q_nope,
            q_rope=inputs.q_rope,
            kv_nope=inputs.kv_cache[:, :, :-64],
            kv_rope=inputs.kv_cache[:, :, -64:],
            req_to_token_indexs=batch.req_to_token_indexs,
            b_req_idx=batch.b_req_idx,
            b_seq_len=batch.b_seq_len,
            softmax_scale=inputs.softmax_scale,
            schedule=batch.schedule,
            out=out,
            alloc_tensor_func=alloc_func,
        )
        return out
