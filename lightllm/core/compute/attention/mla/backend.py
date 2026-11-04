"""MLA kernels consuming replicated latent views and explicit model projections."""

from dataclasses import dataclass
from typing import Callable

import torch

from lightllm.core.compute.attention.dense.backend import BoundAttention
from lightllm.core.contracts.provider import AttentionOffer, ProbeResult
from lightllm.core.compute.attention.mla.spec import MLAAttentionSpec

# Persistent workers consume disjoint sequence splits. Keeping their count fixed
# makes scratch size independent of per-batch autotuning and Graph capture.
DECODE_WORKERS = 128


@dataclass(frozen=True)
class MLAStaticParams:
    spec: MLAAttentionSpec
    alloc_tensor: Callable
    expand_kv: Callable

    @classmethod
    def from_weights(cls, spec, weight, *, device):
        return cls(spec, device.allocate_tensor, weight.cc_kv_b_proj_.mm)


class BoundMLAAttention(BoundAttention):
    def __init__(self, offer, view, params, workspace):
        self.name, self.phase = offer.backend, offer.phase
        self.implementation = self.name.split(".")[0]
        self.spec, self.latent = params.spec, view.fields[0]
        self.alloc_tensor, self.expand_kv = params.alloc_tensor, params.expand_kv
        self.limits, self.workspace = offer.execution_limits, workspace
        self.transfer_requires_compute_sync = offer.transfer_requires_compute_sync
        # Inference uses one-token pages, independent of allocator page geometry.
        self.page_size = 1

    def create_metadata(self, batch):
        if self.phase != "prefill" and batch.max_q_len != 1:
            raise ValueError("MLA decode/verify requires one query row per sequence-length entry")
        if self.phase == "prefill" and (
            batch.total_kv_tokens is None
            or not 0 < batch.total_kv_tokens <= self.limits.max_batch_size * self.limits.max_seq_len
        ):
            raise ValueError("MLA prefill requires the total number of KV tokens")
        return super().create_metadata(batch)

    def run(self, inputs, metadata, out=None):
        if self.phase == "prefill":
            result = self._prefill(inputs, metadata)
        else:
            q_nope, q_rope = inputs
            spec = self.spec
            if (
                q_nope.shape != (metadata.batch_size, spec.q_heads, spec.latent_dim)
                or q_rope.shape != (metadata.batch_size, spec.q_heads, spec.rope_dim)
                or q_nope.dtype != spec.dtype
                or q_rope.dtype != spec.dtype
            ):
                raise ValueError("MLA decode inputs disagree with the bound semantics")
            result = self._decode(q_nope, q_rope, metadata)
        if out is not None:
            out.copy_(result)
            return out
        return result

    def _expanded_kv(self, q, batch):
        from lightllm.core.compute.attention.mla.sample_kv import sample_kv

        spec, total = self.spec, batch.total_kv_tokens
        if q.ndim != 3 or q.shape[1:] != (spec.q_heads, spec.nope_dim + spec.rope_dim) or q.dtype != spec.dtype:
            raise ValueError("MLA prefill input disagrees with the bound semantics")
        latent = self.alloc_tensor((total, 1, spec.latent_dim), spec.dtype)
        k_rope = self.alloc_tensor((total, 1, spec.rope_dim), spec.dtype)
        sample_kv(
            self.latent,
            latent,
            k_rope,
            batch.req_indices,
            batch.req_to_tokens,
            batch.seq_lengths,
            batch.cu_k[:-1],
            batch.max_kv_len,
        )
        expanded = self.alloc_tensor((total, spec.q_heads, spec.nope_dim + spec.value_dim), spec.dtype)
        self.expand_kv(latent.view(total, spec.latent_dim), out=expanded.view(total, -1))
        k_nope, value = expanded.split((spec.nope_dim, spec.value_dim), dim=-1)
        return k_nope, k_rope, value

    def _prefill(self, q, batch):
        spec = self.spec
        k_nope, k_rope, value = self._expanded_kv(q, batch)
        if self.implementation == "fa3":
            from lightllm.utils.sgl_utils import flash_attn_varlen_func

            key = torch.cat((k_nope, k_rope.expand(-1, spec.q_heads, -1)), dim=-1)
            return flash_attn_varlen_func(
                q=q,
                k=key,
                v=value,
                cu_seqlens_q=batch.cu_q,
                cu_seqlens_k=batch.cu_k,
                max_seqlen_q=batch.max_q_len,
                max_seqlen_k=batch.max_kv_len,
                softmax_scale=spec.scale,
                causal=True,
                return_softmax_lse=False,
            )
        from lightllm.common.basemodel.triton_kernel.mla_att.prefill_att import context_attention_fwd_with_v

        out = self.alloc_tensor((q.shape[0], spec.q_heads, spec.value_dim), spec.dtype)
        context_attention_fwd_with_v(
            q[..., : spec.nope_dim],
            q[..., spec.nope_dim :],
            k_nope,
            k_rope,
            value,
            out,
            batch.q_starts,
            batch.cu_k,
            batch.seq_lengths,
            batch.ready_lengths,
            batch.max_q_len,
            spec.scale,
        )
        return out

    def _decode(self, q_nope, q_rope, batch):
        spec = self.spec
        nope, rope = self.latent[..., : spec.latent_dim], self.latent[..., spec.latent_dim :]
        if self.implementation == "fa3":
            from lightllm.utils.sgl_utils import flash_attn_with_kvcache

            return flash_attn_with_kvcache(
                q=q_rope,
                qv=q_nope,
                k_cache=rope.unsqueeze(1),
                v_cache=nope.unsqueeze(1),
                page_table=batch.page_table,
                cache_seqlens=batch.seq_lengths,
                cu_seqlens_q=batch.cu_q,
                cu_seqlens_k_new=batch.cu_k,
                max_seqlen_q=batch.max_q_len,
                softmax_scale=spec.scale,
                causal=True,
                return_softmax_lse=False,
            )
        import triton
        from lightllm.common.basemodel.triton_kernel.mla_att.decode_att.gqa_flash_decoding import (
            _fwd_kernel_calcu_index_and_block_seq,
        )
        from lightllm.common.basemodel.triton_kernel.mla_att.decode_att.gqa_flash_decoding_stage1 import (
            _fwd_kernel_flash_decode_stage1_padding,
        )
        from lightllm.common.basemodel.triton_kernel.mla_att.decode_att.gqa_flash_decoding_stage2 import (
            flash_decode_stage2,
        )

        scratch = self._lane_workspace(batch.microbatch_index)
        index_bytes = (self.limits.max_batch_size + 1) * 8
        indices = scratch[:index_bytes].view(torch.int64)
        block_seq, starts = indices[:1], indices[1 : batch.batch_size + 1]
        slots = DECODE_WORKERS * 4 + self.limits.max_batch_size
        count = spec.q_heads * slots * spec.latent_dim
        values = scratch[index_bytes:].view(torch.float32)
        mid = values[:count].view(spec.q_heads, slots, spec.latent_dim)
        lse = values[count : count + spec.q_heads * slots].view(spec.q_heads, slots)
        _fwd_kernel_calcu_index_and_block_seq[(1,)](
            batch.seq_lengths,
            block_seq,
            starts,
            DECODE_WORKERS,
            batch.batch_size,
            BLOCK_N=16,
            MAX_BATCH_SIZE=triton.next_power_of_2(batch.batch_size),
            num_warps=4,
        )
        _fwd_kernel_flash_decode_stage1_padding[(DECODE_WORKERS,)](
            q_nope,
            q_rope,
            nope,
            rope,
            spec.scale,
            batch.req_to_tokens,
            batch.req_indices,
            batch.seq_lengths,
            mid,
            lse,
            *batch.req_to_tokens.stride(),
            *q_nope.stride(),
            *q_rope.stride(),
            *nope.stride(),
            *rope.stride(),
            *mid.stride(),
            *lse.stride(),
            block_seq,
            num_sm=DECODE_WORKERS,
            head_group_num=triton.cdiv(spec.q_heads, 16),
            head_num=spec.q_heads,
            batch_size=batch.batch_size,
            Q_HEAD_NUM=16,
            BLOCK_DMODEL=spec.latent_dim,
            BLOCK_ROPE_DMODEL=spec.rope_dim,
            BLOCK_N=16,
            NEED_HEAD_MASK=spec.q_heads % 16 != 0,
            NUM_STAGES=2,
            num_warps=4,
            num_stages=1,
        )
        out = self.alloc_tensor(q_nope.shape, spec.dtype)
        flash_decode_stage2(
            block_seq,
            starts,
            mid,
            lse,
            batch.seq_lengths,
            out,
            stage2_num_warps=4,
            stage2_num_stages=2,
        )
        return out


class MLAAttentionBackend:
    bound_class = BoundMLAAttention

    def __init__(self, implementation, phase):
        if implementation not in ("triton", "fa3") or phase not in ("prefill", "decode", "verify"):
            raise ValueError("unsupported MLA implementation or phase")
        self.implementation, self.phase = implementation, phase
        self.name = f"{implementation}.mla.{phase}"

    def probe(self, attention_spec, format_spec, device_caps, limits):
        if not isinstance(attention_spec, MLAAttentionSpec):
            return ProbeResult(reason="requires MLA semantics")
        spec, fmt = attention_spec, format_spec
        if not device_caps.is_nvidia:
            return ProbeResult(reason="MLA requires CUDA")
        if (
            fmt.encoding != "plain"
            or fmt.layout != "token_latent"
            or fmt.fields != ("latent",)
            or fmt.version != 1
            or fmt.dtype not in (torch.float16, torch.bfloat16)
            or spec.dtype != fmt.dtype
        ):
            return ProbeResult(reason="requires plain FP16/BF16 latent and rotary storage")
        if spec.rope_dim != 64 or spec.latent_dim not in (16, 32, 64, 128, 256, 512):
            return ProbeResult(reason="unsupported latent or rotary dimensions")
        if self.implementation == "triton":
            if self.phase == "prefill" and (
                spec.nope_dim != spec.value_dim or spec.nope_dim not in (16, 32, 64, 128, 256, 512)
            ):
                return ProbeResult(reason="Triton prefill requires matching supported non-rotary and value widths")
        else:
            from lightllm.utils.sgl_utils import flash_attn_with_kvcache

            if device_caps.capability is None or device_caps.capability[0] != 9 or flash_attn_with_kvcache is None:
                return ProbeResult(reason="FA3 MLA requires Hopper and installed kernels")
            if self.phase == "prefill" and (spec.nope_dim + spec.rope_dim > 256 or spec.value_dim > 256):
                return ProbeResult(reason="FA3 prefill projection dimensions exceed 256")
            if self.phase != "prefill" and spec.latent_dim < 256:
                return ProbeResult(reason="FA3 MLA decode requires a latent width of at least 256")
        size = 0
        if self.implementation == "fa3":
            size = limits.max_batch_size * limits.max_seq_len * 4
        elif self.phase != "prefill":
            size = (limits.max_batch_size + 1) * 8
            size += spec.q_heads * (DECODE_WORKERS * 4 + limits.max_batch_size) * (spec.latent_dim + 1) * 4
        # Keep lane bases aligned for int64 indices as well as FP32 scratch.
        size = (size + 7) // 8 * 8
        return ProbeResult(
            offers=(
                AttentionOffer(
                    self.name,
                    self.phase,
                    fmt,
                    size * limits.microbatch_count,
                    "serial_layers",
                    limits,
                    transfer_requires_compute_sync=self.implementation == "fa3",
                ),
            )
        )

    def bind(self, offer, state_view, static_params, workspace):
        if offer.backend != self.name or offer.phase != self.phase or offer.format != state_view.format:
            raise ValueError("offer does not match MLA binding")
        (latent,) = state_view.fields
        spec = static_params.spec
        if latent.ndim != 3 or latent.shape[1:] != (1, spec.latent_dim + spec.rope_dim):
            raise ValueError("MLA view shape disagrees with declared projections")
        if latent.dtype != state_view.format.dtype or latent.stride(-1) != 1:
            raise ValueError("MLA view requires contiguous latent elements of the declared dtype")
        if latent.shape[0] % state_view.format.page_size:
            raise ValueError("MLA state capacity must contain complete pages")
        if offer.workspace_bytes and (
            workspace is None
            or workspace.dtype != torch.uint8
            or workspace.numel() < offer.workspace_bytes
            or workspace.device != latent.device
            or not workspace.is_contiguous()
        ):
            raise ValueError("MLA workspace does not satisfy the selected offer")
        return self.bound_class(offer, state_view, static_params, workspace)


from lightllm.core.formats.kv.mla import MLAKVWriter
