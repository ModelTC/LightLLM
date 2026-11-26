"""FlashMLA sparse MLA: prefill over bf16 or packed FP8 caches and decode over packed FP8 caches."""

from dataclasses import dataclass
from functools import lru_cache
from inspect import signature
from typing import Optional

import torch

from lightllm.common.attention.backend import AttentionBackend, TensorAllocator
from lightllm.common.basemodel.infer_struct import InferStateInfo
from . import SparseMLADecodeLayerInputs, SparseMLALayerConfig, SparseMLAPrefillLayerInputs


@dataclass(frozen=True)
class FlashMLASparsePrefillAttState:
    """mem_index holds the cache slots of the step's new tokens; has_cached_prefix says whether any
    request reads keys from earlier steps."""

    mem_index: Optional[torch.Tensor]
    has_cached_prefix: bool


@dataclass(frozen=True)
class FlashMLASparseDecodeAttState:
    """FlashMLA's tile scheduler metadata, filled on the step's first kernel call."""

    sched_meta: object
    uses_cuda_graph: bool = False


@lru_cache(maxsize=2)
def _supports_out(kernel) -> bool:
    """Whether a FlashMLA kernel writes into a caller's out; older builds return their own output."""
    return "out" in signature(kernel).parameters


@lru_cache(maxsize=1)
def _bf16_prefill_out_kernel():
    try:
        from flash_mla import flash_mla_sparse_fwd
    except ImportError:
        return None
    return flash_mla_sparse_fwd if _supports_out(flash_mla_sparse_fwd) else None


def _with_row_axis(indices: torch.Tensor) -> torch.Tensor:
    """FlashMLA reads top-k indices as [rows, 1, topk]."""
    return indices.unsqueeze(1) if indices.ndim == 2 else indices


def _bf16_prefill(q, kv, indices, softmax_scale, out):
    """Use the caller's output directly or bound the legacy kernel's temporary storage to 128 query rows."""
    if q.shape[0] == 0:
        return out
    heads = q.shape[1]
    padded_heads = -(-heads // 64) * 64
    if heads == padded_heads and out.is_contiguous():
        kernel = _bf16_prefill_out_kernel()
        if kernel is not None:
            kernel(q=q, kv=kv, indices=indices, sm_scale=softmax_scale, d_v=out.shape[2], out=out)
            return out

    from sgl_kernel.flash_mla import flash_mla_sparse_fwd

    chunk_size = 128
    for start in range(0, q.shape[0], chunk_size):
        rows = slice(start, start + chunk_size)
        query_chunk = q[rows]
        if heads != padded_heads:
            query_chunk = torch.nn.functional.pad(query_chunk, (0, 0, 0, padded_heads - heads))
        result = flash_mla_sparse_fwd(
            q=query_chunk, kv=kv, indices=indices[rows], sm_scale=softmax_scale, d_v=out.shape[2]
        )[0]
        out[rows].copy_(result[:, :heads])
        del result
    return out


class FlashMLASparsePrefillBackend(
    AttentionBackend[SparseMLALayerConfig, FlashMLASparsePrefillAttState, SparseMLAPrefillLayerInputs]
):
    name = "flashmla_sparse_mla_prefill"

    @classmethod
    def _check_supported(cls, layer_config: SparseMLALayerConfig) -> None:
        if layer_config.kv_cache_format not in ("bf16", "fp8_ds_mla"):
            raise NotImplementedError(f"FlashMLA sparse prefill does not support {layer_config.kv_cache_format!r}")

    def build(self, infer_state: InferStateInfo) -> FlashMLASparsePrefillAttState:
        return FlashMLASparsePrefillAttState(
            mem_index=infer_state.mem_index, has_cached_prefix=(infer_state.max_cache_len or 0) > 0
        )

    def forward(
        self,
        att_state: FlashMLASparsePrefillAttState,
        layer_inputs: SparseMLAPrefillLayerInputs,
        *,
        out: Optional[torch.Tensor] = None,
        alloc_func: TensorAllocator = torch.empty,
    ) -> torch.Tensor:
        """Return [query_tokens, num_heads, kv_lora_rank], filling out if supplied.

        A packed FP8 cache is not read directly: the step's own keys come from current_kv and, when
        some request has a cached prefix, the selected history is dequantized next to them.
        """
        config, q = self.layer_config, layer_inputs.q
        assert q.shape[1] == config.num_heads and q.dtype == config.dtype, "q does not match the layer_config"
        output_shape = (q.shape[0], q.shape[1], config.kv_lora_rank)

        if config.kv_cache_format == "bf16":
            if out is None:
                out = alloc_func(output_shape, dtype=q.dtype, device=q.device)
            indices = _with_row_axis(layer_inputs.topk_mem_indices)
            return _bf16_prefill(q, layer_inputs.kv_cache, indices, config.softmax_scale, out)

        from flash_mla import flash_mla_sparse_fwd

        current_kv = layer_inputs.current_kv
        assert current_kv is not None and layer_inputs.topk_indices is not None
        kv, indices = current_kv, layer_inputs.topk_indices
        if att_state.has_cached_prefix:
            from lightllm.common.basemodel.triton_kernel.kv_copy.prefill_compact_kv_flashmla_fp8 import (
                get_prefill_kv_cache_and_remap_indices_triton,
            )

            kv, indices = get_prefill_kv_cache_and_remap_indices_triton(
                packed_kv=layer_inputs.kv_cache,
                topk_mem_indices=layer_inputs.topk_mem_indices,
                prefill_mem_index=att_state.mem_index,
                prefill_cache_kv=current_kv,
                prefill_dtype=current_kv.dtype,
            )

        supports_out = _supports_out(flash_mla_sparse_fwd)
        if supports_out and out is None:
            out = alloc_func(output_shape, dtype=q.dtype, device=q.device)
        result, _, _ = flash_mla_sparse_fwd(
            q=q,
            kv=kv,
            indices=_with_row_axis(indices),
            sm_scale=config.softmax_scale,
            d_v=config.kv_lora_rank,
            **({"out": out} if supports_out else {}),
        )
        if out is None:
            return result
        if not supports_out:
            out.copy_(result)
        return out


class FlashMLASparseDecodeBackend(
    AttentionBackend[SparseMLALayerConfig, FlashMLASparseDecodeAttState, SparseMLADecodeLayerInputs]
):
    """direct_decode_bound, when set, bounds the valid entries of every top-k row, which lets small
    CUDA graph steps over the "fp8_group64_e8m0" cache use a direct Triton kernel."""

    name = "flashmla_sparse_mla_decode"

    def __init__(self, layer_config: SparseMLALayerConfig) -> None:
        super().__init__(layer_config)
        self.direct_decode_bound = self.group_key(layer_config)
        self._graph_sched_meta: dict[tuple[int, int], object] = {}

    @classmethod
    def _check_supported(cls, layer_config: SparseMLALayerConfig) -> None:
        if layer_config.kv_cache_format not in ("fp8_ds_mla", "fp8_group64_e8m0"):
            raise NotImplementedError("FlashMLA sparse decode supports only the packed FP8 caches")

    @classmethod
    def group_key(cls, layer_config: SparseMLALayerConfig) -> Optional[int]:
        """The direct decode bound: 2048 pooled entries plus at most three kpool remainder entries fit
        the 2112 rounded direct window."""
        if (layer_config.topk, layer_config.kpool) == (2048, 4):
            return layer_config.topk + layer_config.kpool - 1
        return None

    def build(self, infer_state: InferStateInfo) -> FlashMLASparseDecodeAttState:
        """A CUDA graph step keeps the captured scheduler metadata per (microbatch, batch size)."""
        import flash_mla

        if infer_state.uses_cuda_graph:
            key = (infer_state.microbatch_index, infer_state.batch_size)
            if key not in self._graph_sched_meta:
                self._graph_sched_meta[key] = flash_mla.get_mla_metadata()[0]
            sched_meta = self._graph_sched_meta[key]
        else:
            sched_meta = flash_mla.get_mla_metadata()[0]
        return FlashMLASparseDecodeAttState(sched_meta=sched_meta, uses_cuda_graph=infer_state.uses_cuda_graph)

    def forward(
        self,
        att_state: FlashMLASparseDecodeAttState,
        layer_inputs: SparseMLADecodeLayerInputs,
        *,
        out: torch.Tensor,
        alloc_func: TensorAllocator = torch.empty,
    ) -> torch.Tensor:
        """Fill and return out with q_nope's shape; each request has one query row."""
        import flash_mla

        config, q_nope, packed = self.layer_config, layer_inputs.q_nope, layer_inputs.kv_cache
        assert q_nope.shape[1] == config.num_heads and q_nope.dtype == config.dtype, "q does not match the layer_config"
        indices = _with_row_axis(layer_inputs.topk_mem_indices)
        assert indices.shape[1] == 1, "FlashMLA sparse decode expects one query row per request"
        heads = q_nope.shape[1]

        if config.kv_cache_format == "fp8_group64_e8m0":
            from lightllm.common.attention.kernels.sparse_mla import fp8_group64_decode
            from lightllm.common.attention.kernels.sparse_mla.pad_query_heads import pad_query_heads

            if fp8_group64_decode.can_use_fp8_sparse_direct_decode(
                q_nope,
                packed,
                indices,
                config.kv_lora_rank,
                self.direct_decode_bound,
                is_cuda_graph_capture=att_state.uses_cuda_graph,
            ):
                fp8_group64_decode.fp8_sparse_direct_decode(q_nope, packed, indices, config.softmax_scale, out=out)
                return out
            q = pad_query_heads(q_nope, is_cuda_graph_capture=att_state.uses_cuda_graph)
        else:
            q = torch.cat([q_nope, layer_inputs.q_rope], dim=-1).unsqueeze(1).contiguous()

        # One cache slot per page, so the top-k indices address slots directly.
        kv = torch.as_strided(
            packed,
            size=(packed.shape[0], 1, 1, packed.shape[-1]),
            stride=(packed.stride(0), packed.shape[-1], packed.shape[-1], packed.stride(-1)),
        )
        kernel = flash_mla.flash_mla_with_kvcache
        # Padded heads or an older FlashMLA leave the result in the kernel's own output.
        writes_out = q.shape[2] == heads and _supports_out(kernel)
        result, _ = kernel(
            q=q,
            k_cache=kv,
            block_table=None,
            cache_seqlens=None,
            head_dim_v=config.kv_lora_rank,
            tile_scheduler_metadata=att_state.sched_meta,
            softmax_scale=config.softmax_scale,
            causal=False,
            is_fp8_kvcache=True,
            indices=indices,
            **({"out": out.unsqueeze(1)} if writes_out else {}),
        )
        if not writes_out:
            out.copy_(result[:, 0, :heads])
        return out
