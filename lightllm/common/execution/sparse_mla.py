"""Sparse MLA and its FP8 indexer using bound packed State views."""

from dataclasses import dataclass
from importlib import import_module
from importlib.util import find_spec
import inspect

import torch

from .dense import BoundAttention
from .specs import AttentionOffer, ProbeResult, SparseMLAAttentionSpec


def indexer_capacity(limits):
    return (limits.max_batch_size * limits.max_seq_len + 127) // 128 * 128


@dataclass(frozen=True)
class SparseMLAStaticParams:
    spec: SparseMLAAttentionSpec
    alloc_tensor: object


@dataclass
class SparseMLAInputs:
    query: torch.Tensor
    index_query: torch.Tensor
    index_key: torch.Tensor
    index_weights: torch.Tensor
    token_indices: torch.Tensor
    current_kv: torch.Tensor | None = None
    group: object = None


@dataclass
class SparseMLAMetadata:
    batch: object
    ks: torch.Tensor
    ke: torch.Tensor
    lengths: torch.Tensor
    fresh_start: torch.Tensor
    fresh_offset: torch.Tensor
    ragged: torch.Tensor
    selected_lengths: torch.Tensor
    cu_selected: torch.Tensor
    scheduler: object = None

    def copy_for_graph(self, new):
        self.batch.copy_for_graph(new.batch)
        for name in ("ks", "ke", "lengths", "fresh_start", "fresh_offset", "ragged", "selected_lengths", "cu_selected"):
            target, source = getattr(self, name), getattr(new, name)
            if target.shape != source.shape:
                raise ValueError("sparse Graph metadata shape changed")
            if target.data_ptr() != source.data_ptr():
                target.copy_(source, non_blocking=True)


class PackedMLAKVWriter:
    def __init__(self, view):
        self.cache = view.fields[0]
        self.quantized = view.format.encoding == "fp8_group128"

    def __call__(self, *, token_indices, kv):
        if kv.shape != (token_indices.numel(), 1, 576) or kv.dtype != torch.bfloat16:
            raise ValueError("packed MLA writer requires BF16 latent 512 and rotary 64")
        if kv.device != self.cache.device or token_indices.device != self.cache.device:
            raise ValueError("packed MLA inputs must reside on the bound device")
        if not token_indices.numel():
            return
        token_indices = token_indices.contiguous()
        if self.quantized:
            from lightllm.common.basemodel.triton_kernel.kv_copy.destindex_copy_kv_flashmla_fp8 import (
                destindex_copy_kv_flashmla_fp8,
            )

            destindex_copy_kv_flashmla_fp8(
                kv[..., :512],
                kv[..., 512:],
                token_indices,
                self.cache[..., :512].view(torch.float8_e4m3fn),
                self.cache[..., 512:528].view(torch.float32),
                self.cache[..., 528:656].view(torch.bfloat16),
            )
        else:
            from lightllm.common.basemodel.triton_kernel.destindex_copy_kv import destindex_copy_kv

            destindex_copy_kv(K=kv, DestLoc=token_indices, Out=self.cache)


class BoundSparseMLA(BoundAttention):
    flash_mla = None
    requires_dynamic_prefill = False

    def __init__(self, offer, view, params, workspace):
        self.name, self.phase = offer.backend, offer.phase
        self.implementation = self.name.split(".")[0]
        self.spec, self.view = params.spec, view
        self.alloc_tensor = params.alloc_tensor
        self.workspace, self.limits = workspace, offer.execution_limits
        self.transfer_requires_compute_sync = offer.transfer_requires_compute_sync
        self.cache, self.indexer = view.fields
        self.quantized = view.format.encoding == "fp8_group128"
        self.requires_dynamic_prefill = (
            self.quantized and self.implementation == "flashmla_sparse" and self.phase == "prefill"
        )
        self.flash_mla = (
            import_module("flash_mla") if self.quantized and self.implementation == "flashmla_sparse" else None
        )
        if self.flash_mla is not None:
            width = self.cache.shape[-1]
            self.flash_cache = self.cache.as_strided(
                (self.cache.shape[0], 1, 1, width), (self.cache.stride(0), width, width, 1)
            )
        if self.quantized:
            self.payload = self.cache[..., :512].view(torch.float8_e4m3fn)
            self.scales = self.cache[..., 512:528].view(torch.float32)
            self.rope = self.cache[..., 528:656].view(torch.bfloat16)
        else:
            self.payload, self.rope = self.cache[..., :512], self.cache[..., 512:]
            self.scales = self.payload

    @property
    def metadata_key(self):
        heads = (self.spec.q_heads + 63) // 64 * 64 if self.flash_mla is not None else None
        return self.name, self.quantized, heads, self.spec.index_topk

    def create_metadata(self, batch):
        from .sparse_mla_kernels import prepare_sparse_metadata

        if batch.batch_size > self.limits.max_batch_size or batch.max_kv_len > self.limits.max_seq_len:
            raise ValueError("batch exceeds sparse attention capacity")
        if self.phase != "prefill" and batch.max_q_len != 1:
            raise ValueError("sparse decode/verify requires one query row per sequence length")
        count = batch.total_q_tokens if self.phase == "prefill" else batch.batch_size
        if count is None or count <= 0:
            raise ValueError("sparse prefill requires the query token count")
        size = indexer_capacity(self.limits)
        ragged = self._lane_workspace(batch.microbatch_index)[: size * 4].view(torch.int32)
        tensors = [torch.empty(count, dtype=torch.int32, device=self.cache.device) for _ in range(5)]
        prepare_sparse_metadata[(batch.batch_size,)](
            batch.req_to_tokens,
            batch.req_indices,
            batch.seq_lengths,
            batch.cu_q,
            batch.cu_k,
            *tensors,
            ragged,
            batch.req_to_tokens.stride(0),
            256,
        )
        selected = tensors[2].clamp_max(self.spec.index_topk)
        cu_selected = torch.nn.functional.pad(selected.cumsum(0, dtype=torch.int32), (1, 0))
        scheduler = (
            self.flash_mla.get_mla_metadata()[0] if self.flash_mla is not None and self.phase != "prefill" else None
        )
        return SparseMLAMetadata(batch, *tensors, ragged, selected, cu_selected, scheduler)

    def select_indices(self, inputs, metadata):
        from lightllm.models.deepseek3_2.triton_kernel.act_quant import act_quant
        from lightllm.models.deepseek3_2.triton_kernel.destindex_copy_indexer_ks import destindex_copy_indexer_ks
        from sgl_kernel import fast_topk_v2

        spec = self.spec
        q, k, weights = inputs.index_query, inputs.index_key, inputs.index_weights
        tp = self.view.format.shard[1]
        count = inputs.query.shape[0]
        if (
            q.shape != (count, spec.index_heads // tp, 128)
            or k.shape != (count, 128)
            or weights.shape != (count, spec.index_heads)
            or inputs.token_indices.shape != (count,)
        ):
            raise ValueError("indexer inputs disagree with bound sparse semantics")
        if tp > 1:
            from lightllm.distributed import all_gather_into_tensor

            merged = self.alloc_tensor((tp * q.numel(),), q.dtype)
            all_gather_into_tensor(output_=merged, input_=q.reshape(-1), group=inputs.group, async_op=False)
            q = merged.view(tp, count, spec.index_heads // tp, 128).transpose(0, 1).contiguous().view(count, -1, 128)
        q_fp8, q_scale = act_quant(q.contiguous(), spec.index_quant_block, spec.index_scale_fmt)
        k_fp8, k_scale = act_quant(k.contiguous(), spec.index_quant_block, spec.index_scale_fmt)
        destindex_copy_indexer_ks(k_fp8, k_scale, inputs.token_indices.contiguous(), self.indexer)
        weights = weights * q_scale.squeeze(-1)
        logits = self.index_logits(q_fp8, weights, metadata)
        if spec.index_topk == 2048:
            relative = fast_topk_v2(score=logits, lengths=metadata.lengths, topk=spec.index_topk)
        else:
            positions = torch.arange(logits.shape[1], device=logits.device)
            masked = logits.masked_fill(positions[None, :] >= metadata.lengths[:, None], -torch.inf)
            relative = torch.topk(masked, min(spec.index_topk, logits.shape[1]), dim=-1).indices.int()
            relative = torch.where(relative < metadata.lengths[:, None], relative, -1)
            if relative.shape[1] < spec.index_topk:
                relative = torch.nn.functional.pad(relative, (0, spec.index_topk - relative.shape[1]), value=-1)
        logical = torch.where(relative != -1, relative + metadata.ks[:, None], -1)
        physical = torch.where(logical >= 0, metadata.ragged[logical.clamp_min(0).long()], -1)
        return physical, logical

    def index_logits(self, query, weights, metadata):
        from .sparse_mla_kernels import indexer_logits
        import triton

        count = query.shape[0]
        output = self.alloc_tensor((count, self.limits.max_seq_len), torch.float32)
        keys = self.indexer[..., :128].view(torch.float8_e4m3fn)
        scales = self.indexer[..., 128:].view(torch.float32)
        indexer_logits[(count, min(128, triton.cdiv(self.limits.max_seq_len, 64)))](
            query,
            weights.contiguous(),
            keys,
            scales,
            metadata.ks,
            metadata.ke,
            metadata.ragged,
            output,
            keys.stride(0),
            scales.stride(0),
            self.spec.index_heads,
            self.limits.max_seq_len,
            max(16, triton.next_power_of_2(self.spec.index_heads)),
            num_warps=4,
            num_stages=1,
        )
        return output

    def attend(self, query, physical, logical, metadata, current_kv=None):
        if query.shape != (metadata.ks.numel(), self.spec.q_heads, 576) or query.dtype != torch.bfloat16:
            raise ValueError("sparse MLA requires projected BF16 query heads of width 576")
        expected = (query.shape[0], self.spec.index_topk)
        if physical.shape != expected or logical.shape != expected or physical.dtype != torch.int32:
            raise ValueError("sparse indices do not match the declared top-k")
        if self.implementation == "flashmla_sparse":
            if self.quantized:
                return self._flashmla_fp8(query, physical, metadata, current_kv)
            if self.phase == "prefill":
                from sgl_kernel.flash_mla import flash_mla_sparse_fwd

                return flash_mla_sparse_fwd(query, self.cache, physical[:, None], self.spec.scale, 512)[0]
            from lightllm.utils.sgl_utils import flash_attn_with_kvcache

            return flash_attn_with_kvcache(
                q=query[..., 512:],
                qv=query[..., :512],
                k_cache=self.rope.unsqueeze(1),
                v_cache=self.payload.unsqueeze(1),
                page_table=physical,
                cache_seqlens=metadata.selected_lengths,
                cu_seqlens_q=metadata.batch.cu_q.int(),
                cu_seqlens_k_new=metadata.cu_selected,
                max_seqlen_q=1,
                softmax_scale=self.spec.scale,
                causal=False,
                return_softmax_lse=False,
            )
        from .sparse_mla_kernels import indexed_mla
        import triton

        prefill = self.phase == "prefill"
        if (
            self.quantized
            and prefill
            and (
                current_kv is None or current_kv.shape != (query.shape[0], 1, 576) or current_kv.dtype != torch.bfloat16
            )
        ):
            raise ValueError("quantized sparse prefill requires current unquantized KV")
        current_kv = self.cache if current_kv is None else current_kv
        out = self.alloc_tensor((query.shape[0], self.spec.q_heads, 512), torch.bfloat16)
        indexed_mla[(query.shape[0], triton.cdiv(self.spec.q_heads, 16))](
            query,
            self.payload,
            self.scales,
            self.rope,
            physical,
            logical,
            current_kv,
            metadata.fresh_start,
            metadata.fresh_offset,
            out,
            query.stride(0),
            query.stride(1),
            self.payload.stride(0),
            self.scales.stride(0),
            self.rope.stride(0),
            current_kv.stride(0),
            self.spec.q_heads,
            self.spec.index_topk,
            self.spec.scale,
            self.quantized,
            prefill,
            num_warps=8,
            num_stages=1,
        )
        return out

    def _flashmla_fp8(self, query, physical, metadata, current_kv):
        heads = self.spec.q_heads
        if heads % 64:
            query = torch.nn.functional.pad(query, (0, 0, 0, 64 - heads % 64))
        query = query.contiguous()
        if self.phase == "prefill":
            if current_kv is None or current_kv.shape != (query.shape[0], 1, 576) or current_kv.dtype != torch.bfloat16:
                raise ValueError("quantized sparse prefill requires current unquantized KV")
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("FP8 FlashMLA prefill requires a dynamic Graph step for KV compaction")
            from lightllm.common.basemodel.triton_kernel.kv_copy.prefill_compact_kv_flashmla_fp8 import (
                get_prefill_kv_cache_and_remap_indices_triton,
            )

            positions = torch.arange(query.shape[0], device=query.device)
            current_indices = metadata.ragged[(positions - metadata.fresh_offset).long()]
            kv, indices = get_prefill_kv_cache_and_remap_indices_triton(
                self.cache, physical, current_indices, current_kv, current_kv.dtype
            )
            output = self.flash_mla.flash_mla_sparse_fwd(query, kv, indices[:, None], self.spec.scale, 512)[0]
        else:
            # No length hints: the scheduler depends on fixed shapes/top-k, while
            # -1 entries in the current index tensor describe shorter requests.
            output = self.flash_mla.flash_mla_with_kvcache(
                q=query[:, None],
                k_cache=self.flash_cache,
                block_table=None,
                cache_seqlens=None,
                head_dim_v=512,
                tile_scheduler_metadata=metadata.scheduler,
                softmax_scale=self.spec.scale,
                causal=False,
                is_fp8_kvcache=True,
                indices=physical[:, None].contiguous(),
            )[0][:, 0]
        return output[:, :heads]

    def run(self, inputs, metadata, out=None):
        physical, logical = self.select_indices(inputs, metadata)
        result = self.attend(inputs.query, physical, logical, metadata, inputs.current_kv)
        if out is not None:
            out.copy_(result)
            return out
        return result


class SparseMLAAttentionBackend:
    def __init__(self, implementation, phase):
        if implementation not in ("triton", "flashmla_sparse") or phase not in ("prefill", "decode"):
            raise ValueError("unsupported sparse MLA implementation or phase")
        self.implementation, self.phase = implementation, phase
        self.name = f"{implementation}.sparse_mla.{phase}"

    def probe(self, attention_spec, format_spec, device_caps, limits):
        if not isinstance(attention_spec, SparseMLAAttentionSpec):
            return ProbeResult(reason="requires sparse MLA semantics")
        spec, fmt = attention_spec, format_spec
        if device_caps.device_type != "cuda" or device_caps.capability is None or device_caps.capability[0] != 9:
            return ProbeResult(reason="the FP8 sparse indexer requires Hopper")
        if find_spec("sgl_kernel") is None:
            return ProbeResult(reason="requires SGL top-k kernels")
        if (
            spec.dtype != torch.bfloat16
            or (spec.latent_dim, spec.rope_dim, spec.index_dim) != (512, 64, 128)
            or spec.index_quant_block != 128
            or spec.index_heads % fmt.shard[1]
        ):
            return ProbeResult(reason="requires BF16 latent 512, rotary 64 and whole 128-wide indexer heads")
        expected_dtype = torch.bfloat16 if fmt.encoding == "plain" else torch.uint8
        if (
            fmt.layout != "token_packed_mla"
            or fmt.encoding not in ("plain", "fp8_group128")
            or fmt.dtype != expected_dtype
            or fmt.fields != ("latent", "indexer")
            or fmt.version != 1
            or fmt.field_dtypes != (expected_dtype, torch.uint8)
        ):
            return ProbeResult(reason="requires the declared packed latent and FP8 indexer fields")
        if fmt.encoding == "fp8_group128" and (fmt.group_size != 128 or fmt.scale_dtype != torch.float32):
            return ProbeResult(reason="packed FP8 requires group128 and FP32 power-of-two scales")
        if self.implementation == "flashmla_sparse":
            if fmt.encoding == "fp8_group128":
                if spec.q_heads > 128 or spec.index_topk % 256:
                    return ProbeResult(reason="FP8 FlashMLA requires at most 128 query heads and 256-top-k alignment")
                if find_spec("flash_mla") is None:
                    return ProbeResult(reason="packed FP8 FlashMLA requires the standalone flash_mla package")
                try:
                    module = import_module("flash_mla")
                    inspect.signature(module.get_mla_metadata).bind()
                    required = {"indices", "is_fp8_kvcache", "tile_scheduler_metadata"}
                    if not required.issubset(inspect.signature(module.flash_mla_with_kvcache).parameters):
                        raise ValueError("missing sparse decode arguments")
                    if not callable(module.flash_mla_sparse_fwd):
                        raise ValueError("missing sparse prefill")
                except (ImportError, OSError, AttributeError, TypeError, ValueError) as error:
                    return ProbeResult(reason=f"incompatible standalone FlashMLA API: {error}")
            elif self.phase == "prefill" and (spec.q_heads % 64 or spec.index_topk % 256):
                return ProbeResult(reason="FlashMLA prefill requires BF16 storage, 64-head and 256-top-k alignment")
        size = indexer_capacity(limits) * 4
        return ProbeResult(
            offers=(
                AttentionOffer(
                    self.name,
                    self.phase,
                    fmt,
                    size * limits.microbatch_count,
                    "serial_layers",
                    limits,
                    transfer_requires_compute_sync=self.implementation == "flashmla_sparse",
                ),
            )
        )

    def bind(self, offer, state_view, static_params, workspace):
        if offer.backend != self.name or offer.phase != self.phase or offer.format != state_view.format:
            raise ValueError("offer does not match sparse MLA binding")
        kv, indexer = state_view.fields
        width = 576 if state_view.format.encoding == "plain" else 656
        if (
            kv.ndim != 3
            or kv.shape[1:] != (1, width)
            or kv.dtype != state_view.format.dtype
            or indexer.shape != (kv.shape[0], 1, 132)
            or indexer.dtype != torch.uint8
            or indexer.device != kv.device
            or kv.stride(-1) != 1
            or indexer.stride(-1) != 1
        ):
            raise ValueError("sparse MLA fields disagree with the packed format")
        if (
            workspace is None
            or workspace.numel() < offer.workspace_bytes
            or workspace.dtype != torch.uint8
            or workspace.device != kv.device
            or not workspace.is_contiguous()
        ):
            raise ValueError("sparse MLA workspace does not satisfy the selected offer")
        return BoundSparseMLA(offer, state_view, static_params, workspace)
