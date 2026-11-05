"""Pooled NoPE attention consumes history and a separately retained indexer tail."""

from lightllm.common.execution.compute.attention.workspace import lane_workspace


from typing import Callable
from lightllm.common.execution.compute.attention.batch import AttentionBatch

from dataclasses import dataclass, replace
from importlib import import_module
from importlib.util import find_spec
import inspect

import torch
import triton

from lightllm.common.execution.compute.attention.sparse_mla.backend import BoundSparseMLA, indexer_capacity
from lightllm.common.execution.contracts.provider import AttentionOffer, ProbeResult
from lightllm.common.execution.compute.attention.pooled_nope.spec import PooledNoPEAttentionSpec


def fp8_scratch_fields(spec, phase, limits):
    fields = {}
    if spec.index_topk != 2048:
        return fields
    heads = spec.q_heads
    if phase == "prefill":
        if indexer_capacity(limits) < 65536 or heads not in (8, 64):
            return fields
        batch, rows = (8, 136) if heads == 8 else (1, 34)
        if heads == 64:
            fields.update(
                compact=((8, 2176), torch.int64),
                lengths=((8,), torch.int32),
                page_table=((8, 2176), torch.int32),
                cu_q=((9,), torch.int32),
                cu_k=((9,), torch.int32),
                selected_kv=((8 * 2176, 1, 512), torch.bfloat16),
            )
    else:
        batch = min(limits.max_batch_size, 8 // triton.cdiv(heads, 16))
        if not batch:
            return fields
        rows = max(n * (132 if n == 1 and heads <= 16 else 33) for n in range(1, batch + 1))
    fields.update(
        partial_o=((rows, heads, 512), torch.float32),
        partial_lse=((rows, heads), torch.float32),
        output=((batch, heads, 512), torch.bfloat16),
    )
    return fields


def fp8_scratch_bytes(fields):
    from math import prod

    return sum((prod(shape) * dtype.itemsize + 7) // 8 * 8 for shape, dtype in fields.values())


@dataclass(frozen=True)
class PooledNoPEStaticParams:
    spec: PooledNoPEAttentionSpec
    alloc_tensor: Callable
    pool_position_bias: torch.Tensor


@dataclass(frozen=True)
class PooledNoPEInputs:
    query: torch.Tensor
    index_key: torch.Tensor
    index_gate: torch.Tensor
    index_query: torch.Tensor | None
    index_weights: torch.Tensor | None
    current_kv: torch.Tensor | None = None


class BoundPooledNoPE(BoundSparseMLA):
    def __init__(self, offer, view, params, workspace, tail, *, implementation):
        self.name, self.phase = offer.backend, offer.phase
        self.implementation = implementation
        self.spec, self.view = params.spec, view
        self.alloc_tensor = params.alloc_tensor
        self.pool_position_bias = params.pool_position_bias
        self.workspace, self.limits = workspace, offer.execution_limits
        self.transfer_requires_compute_sync = offer.transfer_requires_compute_sync
        self.cache, self.indexer = view.fields
        self.indexer = self.indexer.view(-1, 132)
        self.tail = tail
        self.quantized = view.format.encoding == "fp8_group64_e8m0"
        self.fp8_scratch = {}
        if self.quantized and self.implementation == "flashmla_sparse":
            from math import prod

            if self.phase != "prefill":
                self.flash_mla = import_module("flash_mla")
                self.flash_cache = self.cache.as_strided(
                    (self.cache.shape[0], 1, 1, 584), (self.cache.stride(0), 584, 584, 1)
                )
            for lane in range(self.limits.microbatch_count):
                buffer = lane_workspace(self.workspace, self.limits, lane)
                offset, fields = indexer_capacity(self.limits) * 4, {}
                for name, (shape, dtype) in fp8_scratch_fields(self.spec, self.phase, self.limits).items():
                    size = prod(shape) * dtype.itemsize
                    fields[name] = buffer[offset : offset + size].view(dtype).view(shape)
                    offset += (size + 7) // 8 * 8
                self.fp8_scratch[lane] = fields

    @property
    def metadata_key(self):
        return self.name, self.spec, self.quantized

    def create_metadata(self, batch):
        if self.phase == "verify" and batch.mtp_indices is None:
            raise ValueError("pooled verification requires explicit candidate indices")
        if self.quantized and self.implementation == "flashmla_sparse" and self.phase == "prefill":
            if batch.total_kv_tokens is None or not 0 < batch.total_kv_tokens <= indexer_capacity(self.limits):
                raise ValueError("FP8 pooled prefill requires the bounded total KV token count")
            # HOLD spans leave gaps in CU_K. Gather may visit those rows even
            # though no attention index selects them; use the reserved HOLD row.
            lane_workspace(self.workspace, self.limits, batch.microbatch_index)[: batch.total_kv_tokens * 4].view(
                torch.int32
            ).fill_(self.cache.shape[0] - 1)
        # HOLD consumes no history even when the runner pads it with query tokens.
        safe_batch = replace(
            batch, seq_lengths=batch.seq_lengths.masked_fill(batch.req_indices == self.tail.shape[0] - 1, 0)
        )
        metadata = super().create_metadata(safe_batch)
        lengths, pool = metadata.lengths, self.spec.pool_size
        metadata.selected_lengths = (lengths // pool * pool).clamp_max(self.spec.index_topk) + lengths % pool
        metadata.cu_selected = torch.nn.functional.pad(metadata.selected_lengths.cumsum(0, dtype=torch.int32), (1, 0))
        return metadata

    def _fp8_attention(self, inputs, physical, relative, metadata):
        q, spec = inputs.query, self.spec
        fields = self.fp8_scratch[metadata.batch.microbatch_index]
        batch = q.shape[0]
        if self.phase == "prefill":
            from lightllm.common.execution.compute.attention.pooled_nope.prefill_fp8 import context_attention_fwd_fp8

            fresh = inputs.current_kv
            if fresh is None or fresh.shape != (batch, 1, 512) or fresh.dtype != spec.dtype:
                raise ValueError("quantized prefill requires current unquantized KV")
            positions = torch.arange(batch, device=q.device) - metadata.fresh_offset
            current_indices = metadata.ragged[positions.clamp_min(0).long()]
            current_indices = current_indices.masked_fill(metadata.lengths == 0, self.cache.shape[0] - 1)
            direct = selected = None
            if fields and ((spec.q_heads == 8 and batch <= 8) or (spec.q_heads == 64 and batch == 1)):
                splits = 34 if batch == 1 else 17
                direct = self._direct_workspace(fields, batch, splits)
            if "compact" in fields and 2 <= batch <= 8:
                selected = {name: fields[name][:batch] for name in ("compact", "lengths", "page_table")}
                selected.update(
                    cu_q=fields["cu_q"][: batch + 1],
                    cu_k=fields["cu_k"][: batch + 1],
                    selected_kv=fields["selected_kv"][: batch * 2176],
                )
            output = context_attention_fwd_fp8(
                q,
                self.cache,
                metadata.ragged[: metadata.batch.total_kv_tokens],
                current_indices,
                fresh,
                metadata.ks,
                relative,
                spec.score_scale,
                selected_workspace=selected,
                direct_workspace=direct,
            )
        else:
            from lightllm.common.execution.compute.attention.pooled_nope.decode_fp8 import (
                can_use_fp8_sparse_direct_decode,
                fp8_sparse_direct_decode,
            )

            indices = physical[:, None]
            if can_use_fp8_sparse_direct_decode(
                q,
                self.cache,
                indices,
                512,
                2051 if spec.index_topk == 2048 else None,
                is_cuda_graph_capture=torch.cuda.is_current_stream_capturing(),
            ):
                splits = 132 if batch == 1 and spec.q_heads <= 16 else 33
                output = fp8_sparse_direct_decode(
                    q, self.cache, indices, spec.score_scale, workspace=self._direct_workspace(fields, batch, splits)
                )
            else:
                padded = torch.nn.functional.pad(q, (0, 0, 0, (-spec.q_heads) % 64)).contiguous()
                output = self.flash_mla.flash_mla_with_kvcache(
                    q=padded[:, None],
                    k_cache=self.flash_cache,
                    block_table=None,
                    cache_seqlens=None,
                    head_dim_v=512,
                    tile_scheduler_metadata=metadata.scheduler,
                    softmax_scale=spec.score_scale,
                    causal=False,
                    is_fp8_kvcache=True,
                    indices=indices.contiguous(),
                )[0][:, 0, : spec.q_heads]
        return output.masked_fill((metadata.lengths == 0)[:, None, None], 0)

    def _direct_workspace(self, fields, batch, splits):
        return {
            "partial_o": fields["partial_o"][: batch * splits].view(batch, splits, self.spec.q_heads, 512),
            "partial_lse": fields["partial_lse"][: batch * splits].view(batch, splits, self.spec.q_heads),
            "output": fields["output"][:batch],
        }

    def requires_indexer(self, max_kv_len):
        # Decode Graphs may grow across top-k; select their path from the plan's
        # capacity, not the length of the batch that happened to be captured.
        capacity = self.limits.max_seq_len if self.phase != "prefill" else max_kv_len
        return capacity > self.spec.index_topk

    def select_indices(self, inputs, metadata):
        from lightllm.common.execution.compute.attention.pooled_nope.kpool import compress_pools, expand_topk
        from lightllm.common.execution.formats.kv.kernels.index_quant import hadamard_transform_quant_fp8
        from lightllm.common.execution.compute.attention.sparse_mla.kernels import indexer_logits

        spec, batch = self.spec, metadata.batch
        count = inputs.query.shape[0]
        for tensor, shape in (
            (inputs.index_key, (count, spec.index_dim)),
            (inputs.index_gate, (count, spec.index_dim)),
        ):
            if tensor.shape != shape or tensor.dtype != spec.dtype or tensor.device != self.cache.device:
                raise ValueError("pooled indexer key/gate disagree with the bound semantics")
        raw = torch.cat((inputs.index_key, inputs.index_gate), -1)
        compress_pools(
            raw,
            self.tail,
            self.indexer,
            self.pool_position_bias,
            metadata.lengths,
            metadata.ks,
            metadata.ragged,
            batch.req_indices,
            batch.cu_q,
            batch.seq_lengths,
            batch.max_q_len,
            mtp_index=batch.mtp_indices if self.phase != "prefill" else None,
        )
        if not self.requires_indexer(batch.max_kv_len):
            return expand_topk(None, metadata.lengths, metadata.ks, metadata.ragged, spec.index_topk, dense=True)
        if inputs.index_query is None or inputs.index_weights is None:
            raise ValueError("sparse pool selection requires explicit indexer queries and weights")
        q, weights = inputs.index_query, inputs.index_weights
        if q.shape != (count, spec.index_heads, spec.index_dim) or weights.shape != (count, spec.index_heads):
            raise ValueError("pooled indexer query/weights disagree with the bound semantics")
        q_fp8, q_scale = hadamard_transform_quant_fp8(q, scale=spec.index_dim ** -0.5)
        weights = weights * (spec.index_heads ** -0.5 * spec.index_dim ** -0.5) * q_scale.squeeze(-1)
        width = triton.cdiv(self.limits.max_seq_len, spec.pool_size)
        logits = self.alloc_tensor((count, width), torch.float32)
        keys = self.indexer[:, :128].view(torch.float8_e4m3fn)
        scales = self.indexer[:, 128:].view(torch.float32)
        indexer_logits[(count, min(triton.cdiv(width, 64), 128))](
            q_fp8,
            weights.contiguous(),
            keys,
            scales,
            metadata.ks,
            metadata.ke,
            metadata.ragged,
            logits,
            keys.stride(0),
            scales.stride(0),
            spec.index_heads,
            width,
            max(16, triton.next_power_of_2(spec.index_heads)),
            POOL_SIZE=spec.pool_size,
        )
        lengths = metadata.lengths // spec.pool_size
        positions = torch.arange(width, device=logits.device)
        logits.masked_fill_(positions[None, :] >= lengths[:, None], -torch.inf)
        take = min(width, spec.index_topk // spec.pool_size)
        groups = torch.topk(logits, take, dim=-1).indices.int()
        groups.masked_fill_(groups >= lengths[:, None], -1)
        if take < spec.index_topk // spec.pool_size:
            groups = torch.nn.functional.pad(groups, (0, spec.index_topk // spec.pool_size - take), value=-1)
        return expand_topk(groups, metadata.lengths, metadata.ks, metadata.ragged, spec.index_topk)

    def run(self, inputs, metadata, out=None):
        from lightllm.common.execution.compute.attention.sparse_mla.kernels import indexed_nope

        q, spec = inputs.query, self.spec
        if q.shape[1:] != (spec.q_heads, spec.latent_dim) or q.dtype != spec.dtype or q.device != self.cache.device:
            raise ValueError("pooled NoPE query disagrees with the bound semantics")
        physical, relative = self.select_indices(inputs, metadata)
        if self.implementation == "flashmla_sparse":
            if self.quantized:
                result = self._fp8_attention(inputs, physical, relative, metadata)
            elif self.phase == "prefill":
                from sgl_kernel.flash_mla import flash_mla_sparse_fwd

                padded_heads = triton.cdiv(spec.q_heads, 64) * 64
                padded = (
                    q
                    if padded_heads == spec.q_heads
                    else torch.nn.functional.pad(q, (0, 0, 0, padded_heads - spec.q_heads))
                )
                result, _, _ = flash_mla_sparse_fwd(
                    q=padded,
                    kv=self.cache,
                    indices=physical.unsqueeze(1),
                    sm_scale=spec.score_scale,
                    d_v=spec.latent_dim,
                )
                result = result[:, : spec.q_heads]
            else:
                from sgl_kernel.flash_attn import flash_attn_with_kvcache

                result = flash_attn_with_kvcache(
                    q=None,
                    qv=q,
                    k_cache=None,
                    v_cache=self.cache.unsqueeze(1),
                    page_table=physical,
                    cache_seqlens=metadata.selected_lengths,
                    cu_seqlens_q=metadata.batch.cu_q,
                    cu_seqlens_k_new=metadata.cu_selected,
                    max_seqlen_q=1,
                    softmax_scale=spec.score_scale,
                    causal=False,
                    only_qv=True,
                )
            if out is not None:
                out.copy_(result)
                return out
            return result
        if out is None:
            out = self.alloc_tensor(tuple(q.shape), q.dtype)
        if out.shape != q.shape or out.dtype != q.dtype or out.device != q.device or not out.is_contiguous():
            raise ValueError("pooled NoPE output must match the query layout")
        fresh = inputs.current_kv
        if self.quantized and self.phase == "prefill":
            if fresh is None or fresh.shape != (q.shape[0], 1, spec.latent_dim) or fresh.dtype != q.dtype:
                raise ValueError("quantized prefill requires current unquantized KV")
        logical = torch.where(relative >= 0, relative + metadata.ks[:, None], -1)
        indexed_nope[(q.shape[0], triton.cdiv(spec.q_heads, 16))](
            q,
            self.cache,
            physical,
            logical,
            fresh,
            metadata.fresh_start,
            metadata.fresh_offset,
            out,
            q.stride(0),
            q.stride(1),
            self.cache.stride(0),
            fresh.stride(0) if fresh is not None else 0,
            spec.q_heads,
            spec.latent_dim,
            physical.shape[1],
            spec.score_scale,
            self.quantized,
            self.phase == "prefill",
            num_warps=4,
            num_stages=1,
        )
        return out


class PooledNoPEAttentionBackend:
    supports_auxiliary_states = True

    def __init__(self, implementation, phase):
        if implementation not in ("triton", "flashmla_sparse") or phase not in ("prefill", "decode", "verify"):
            raise ValueError("unsupported pooled NoPE implementation or phase")
        self.implementation, self.phase = implementation, phase
        self.name = f"{implementation}.pooled_nope.{phase}"

    def probe(self, spec, fmt, device_caps, limits, auxiliary_formats=None):
        if not isinstance(spec, PooledNoPEAttentionSpec):
            return ProbeResult(reason="requires pooled NoPE semantics")
        if (
            not device_caps.is_nvidia
            or spec.dtype not in (torch.float16, torch.bfloat16)
            or spec.latent_dim != 512
            or spec.index_dim != 128
            or spec.pool_size != 4
        ):
            return ProbeResult(reason="pooled NoPE requires CUDA, latent 512, indexer 128 and pool size 4")
        tail = (auxiliary_formats or {}).get("indexer_tail")
        if (
            set(auxiliary_formats or {}) != {"indexer_tail"}
            or tail.encoding != "plain"
            or tail.layout != "request_indexer_window"
            or tail.fields != ("tail",)
            or tail.dtype != spec.dtype
            or tail.shard != fmt.shard
            or tail.version != 1
        ):
            return ProbeResult(reason="pooled NoPE requires a compatible independent indexer tail")
        if tail.candidate_slots < limits.max_verify_tokens or (
            self.phase == "verify" and limits.max_verify_tokens == 1
        ):
            return ProbeResult(reason="pooled verification requires enough retained tail candidate slots")
        quantized = fmt.encoding == "fp8_group64_e8m0"
        if (
            fmt.encoding not in ("plain", "fp8_group64_e8m0")
            or fmt.layout != "token_pooled_nope"
            or fmt.fields != ("latent", "indexer")
            or fmt.version != 1
            or fmt.dtype != (torch.uint8 if quantized else spec.dtype)
            or fmt.field_dtypes != (fmt.dtype, torch.uint8)
            or (quantized and (spec.dtype != torch.bfloat16 or fmt.group_size != 64 or fmt.scale_dtype != torch.uint8))
        ):
            return ProbeResult(reason="unsupported pooled latent/indexer format")
        if self.implementation == "flashmla_sparse":
            if self.phase == "prefill" and spec.dtype != torch.bfloat16:
                return ProbeResult(reason="SGL sparse NoPE prefill requires BF16 inputs")
            if find_spec("sgl_kernel") is None or device_caps.capability is None or device_caps.capability[0] < 9:
                return ProbeResult(reason="SGL pooled NoPE requires Hopper or newer and installed kernels")
            if quantized and self.phase != "prefill":
                if spec.q_heads > 128 or find_spec("flash_mla") is None:
                    return ProbeResult(reason="FP8 pooled decode requires standalone FlashMLA and at most 128 heads")
                try:
                    module = import_module("flash_mla")
                    inspect.signature(module.get_mla_metadata).bind()
                    required = {"indices", "is_fp8_kvcache", "tile_scheduler_metadata"}
                    if not required.issubset(inspect.signature(module.flash_mla_with_kvcache).parameters):
                        raise ValueError("missing sparse decode arguments")
                except (ImportError, OSError, AttributeError, TypeError, ValueError) as error:
                    return ProbeResult(reason=f"incompatible standalone FlashMLA API: {error}")
            elif self.phase != "prefill" or quantized:
                from sgl_kernel.flash_attn import flash_attn_with_kvcache

                if "only_qv" not in inspect.signature(flash_attn_with_kvcache).parameters:
                    return ProbeResult(reason="installed SGL decode does not support NoPE only_qv")
        size = indexer_capacity(limits) * 4 * limits.microbatch_count
        if quantized and self.implementation == "flashmla_sparse":
            size += fp8_scratch_bytes(fp8_scratch_fields(spec, self.phase, limits)) * limits.microbatch_count
        return ProbeResult(offers=(AttentionOffer(self.name, self.phase, fmt, size, "pooled_nope", limits),))

    def bind(self, offer, view, params, workspace, *, auxiliary_views):
        if offer.backend != self.name or offer.phase != self.phase or offer.format != view.format:
            raise ValueError("offer does not match pooled NoPE binding")
        spec, cache, indexer = params.spec, *view.fields
        if (
            workspace is None
            or workspace.dtype != torch.uint8
            or not workspace.is_contiguous()
            or workspace.device != cache.device
            or workspace.numel() < offer.workspace_bytes
        ):
            raise ValueError("pooled NoPE workspace does not satisfy the selected offer")
        tail = auxiliary_views["indexer_tail"].fields[0]
        width = 584 if view.format.encoding == "fp8_group64_e8m0" else spec.latent_dim
        if (
            cache.ndim != 3
            or cache.shape[1:] != (1, width)
            or cache.dtype != view.format.dtype
            or indexer.shape != (cache.shape[0], 1, 132)
            or indexer.dtype != torch.uint8
            or tail.ndim != 3
            or tail.shape[1:]
            != (spec.pool_size + auxiliary_views["indexer_tail"].format.candidate_slots - 1, 2 * spec.index_dim)
            or tail.dtype != spec.dtype
            or not tail.is_contiguous()
            or cache.stride(-1) != 1
            or indexer.stride(-1) != 1
            or tail.device != cache.device
            or indexer.device != cache.device
            or params.pool_position_bias.shape != (spec.pool_size, spec.index_dim)
            or params.pool_position_bias.dtype != torch.float32
            or params.pool_position_bias.device != cache.device
            or not params.pool_position_bias.is_contiguous()
        ):
            raise ValueError("pooled NoPE views or static pool parameters disagree with semantics")
        return BoundPooledNoPE(offer, view, params, workspace, tail, implementation=self.implementation)
