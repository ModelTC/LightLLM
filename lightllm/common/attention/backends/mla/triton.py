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
    def plan(
        self,
        *,
        b_q_start_loc: torch.Tensor,
        b_kv_start_loc: torch.Tensor,
        b_seq_len: torch.Tensor,
        b_ready_cache_len: torch.Tensor,
        max_q_seq_len: int,
    ) -> TritonMLAPrefillBatch:
        """The kernel reads the metadata tensors directly, so planning only records them."""
        return TritonMLAPrefillBatch(
            b_q_start_loc=b_q_start_loc,
            b_kv_start_loc=b_kv_start_loc,
            b_seq_len=b_seq_len,
            b_ready_cache_len=b_ready_cache_len,
            max_q_seq_len=max_q_seq_len,
        )

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

    kv_cache is a borrowed [physical_slots, 1, latent_dim + rope_dim] cache. Its latent
    prefix supplies both K and V. The cache includes the current token for each request.
    """

    q_nope: torch.Tensor
    q_rope: torch.Tensor
    kv_cache: torch.Tensor
    softmax_scale: float


@dataclass(frozen=True)
class TritonMLADecodeBatch:
    """At decode, request i reads physical slots req_to_token_indexs[b_req_idx[i], :b_seq_len[i]].

    Built by TritonMLADecodeBackend.plan and shared by all layers of one decode batch.
    """

    req_to_token_indexs: torch.Tensor
    b_req_idx: torch.Tensor
    b_seq_len: torch.Tensor
    schedule: "MLADecodeSchedule"


class TritonMLADecodeBackend(AttentionBackend[TritonMLADecodeBatch, MLADecodeInputs]):
    def __init__(self, *, q_heads: int, latent_dim: int, rope_dim: int, dtype: torch.dtype) -> None:
        """Fix the per-rank layer configuration so batches can be planned before any layer runs."""
        self.q_heads = q_heads
        self.latent_dim = latent_dim
        self.rope_dim = rope_dim
        self.dtype = dtype

    def plan(
        self,
        *,
        req_to_token_indexs: torch.Tensor,
        b_req_idx: torch.Tensor,
        b_seq_len: torch.Tensor,
        tuning_seq_len: int,
    ) -> TritonMLADecodeBatch:
        """Plan the split schedule from the current b_seq_len, outside CUDA graph capture.

        tuning_seq_len is a host-side configuration hint, independent of the actual lengths.
        A batch reused by CUDA graph replay must be refreshed after b_seq_len changes.
        """
        from lightllm.common.attention.kernels.mla.decode.gqa_flash_decoding import plan_mla_decode_schedule

        schedule = plan_mla_decode_schedule(
            b_seq_len=b_seq_len,
            q_head_num=self.q_heads,
            kv_lora_rank=self.latent_dim,
            q_rope_dim=self.rope_dim,
            dtype=self.dtype,
            tuning_seq_len=tuning_seq_len,
        )
        return TritonMLADecodeBatch(
            req_to_token_indexs=req_to_token_indexs, b_req_idx=b_req_idx, b_seq_len=b_seq_len, schedule=schedule
        )

    def refresh(self, batch: TritonMLADecodeBatch) -> None:
        """Recompute splits in place after the caller rewrites batch.b_seq_len, e.g. before graph replay."""
        from lightllm.common.attention.kernels.mla.decode.gqa_flash_decoding import update_mla_decode_schedule

        update_mla_decode_schedule(batch.schedule, batch.b_seq_len)

    def forward(
        self,
        batch: TritonMLADecodeBatch,
        inputs: MLADecodeInputs,
        *,
        out: torch.Tensor,
        alloc_func: TensorAllocator = torch.empty,
    ) -> torch.Tensor:
        """Fill and return out with q_nope's [requests, heads, latent_dim] shape, dtype and device.

        Inputs must match the configured heads, latent/rotary dims and dtype.
        Inputs and cache are read-only. out has unit last stride and must not overlap them.
        """
        from lightllm.common.attention.kernels.mla.decode.gqa_flash_decoding import (
            gqa_token_decode_attention_flash_decoding,
        )

        q_nope, q_rope = inputs.q_nope, inputs.q_rope
        assert q_nope.shape[1:] == (self.q_heads, self.latent_dim) and q_nope.dtype == self.dtype
        assert q_rope.shape[-1] == self.rope_dim
        assert inputs.kv_cache.shape[-1] == self.latent_dim + self.rope_dim
        gqa_token_decode_attention_flash_decoding(
            q_nope=q_nope,
            q_rope=q_rope,
            kv_nope=inputs.kv_cache[:, :, : self.latent_dim],
            kv_rope=inputs.kv_cache[:, :, self.latent_dim :],
            req_to_token_indexs=batch.req_to_token_indexs,
            b_req_idx=batch.b_req_idx,
            b_seq_len=batch.b_seq_len,
            softmax_scale=inputs.softmax_scale,
            schedule=batch.schedule,
            out=out,
            alloc_tensor_func=alloc_func,
        )
        return out
