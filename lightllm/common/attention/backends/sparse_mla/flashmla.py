"""FlashMLA sparse MLA: prefill over bf16 or packed FP8 caches and decode over packed FP8 caches."""

from dataclasses import dataclass
from functools import lru_cache
from inspect import signature
from typing import Optional

import torch

from lightllm.common.attention.backend import (
    AttentionBackend,
    AttentionBuildResources,
    TensorAllocator,
)
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
    """FlashMLA's tile scheduler metadata, filled on the step's first kernel call.

    direct_decode_bound, when set, bounds the valid entries of every top-k row, which lets small
    CUDA graph steps over the "fp8_group64_e8m0" cache use a direct Triton kernel.
    """

    sched_meta: object
    uses_cuda_graph: bool = False
    direct_decode_bound: Optional[int] = None


class FlashMLASparseDecodeBuildContext:
    def __init__(self, *, direct_decode_bound: Optional[int] = None) -> None:
        self.direct_decode_bound = direct_decode_bound
        self._cuda_graph_sched_meta: dict[tuple[int, int], object] = {}


def _check_inputs(layer_config: SparseMLALayerConfig, q: torch.Tensor) -> None:
    assert q.shape[1] == layer_config.num_heads and q.dtype == layer_config.dtype, "q does not match the layer_config"


@lru_cache(maxsize=1)
def _bf16_prefill_out_kernel():
    try:
        from flash_mla import flash_mla_sparse_fwd
    except ImportError:
        return None
    return flash_mla_sparse_fwd if "out" in signature(flash_mla_sparse_fwd).parameters else None


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
    def __init__(self, layer_config: SparseMLALayerConfig) -> None:
        super().__init__(layer_config)
        if layer_config.kv_cache_format not in ("bf16", "fp8_ds_mla"):
            raise NotImplementedError(f"FlashMLA sparse prefill does not support {layer_config.kv_cache_format!r}")

    def forward(
        self,
        att_state: FlashMLASparsePrefillAttState,
        layer_inputs: SparseMLAPrefillLayerInputs,
        *,
        out: torch.Tensor,
        alloc_func: TensorAllocator = torch.empty,
    ) -> torch.Tensor:
        """Fill and return out [query_tokens, num_heads, kv_lora_rank].

        A packed FP8 cache is not read directly: the step's own keys come from current_kv and, when
        some request has a cached prefix, the selected history is dequantized next to them.
        """
        layer_config, q = self.layer_config, layer_inputs.q
        _check_inputs(layer_config, q)
        if layer_config.kv_cache_format == "bf16":
            kv, indices = layer_inputs.kv_cache, layer_inputs.topk_mem_indices
            if indices.ndim == 2:
                indices = indices.unsqueeze(1)
            return _bf16_prefill(q, kv, indices, layer_config.softmax_scale, out)

        from flash_mla import flash_mla_sparse_fwd

        assert layer_inputs.current_kv is not None and layer_inputs.topk_indices is not None
        kv, indices = layer_inputs.current_kv, layer_inputs.topk_indices
        if att_state.has_cached_prefix:
            from lightllm.common.basemodel.triton_kernel.kv_copy.prefill_compact_kv_flashmla_fp8 import (
                get_prefill_kv_cache_and_remap_indices_triton,
            )

            kv, indices = get_prefill_kv_cache_and_remap_indices_triton(
                packed_kv=layer_inputs.kv_cache,
                topk_mem_indices=layer_inputs.topk_mem_indices,
                prefill_mem_index=att_state.mem_index,
                prefill_cache_kv=layer_inputs.current_kv,
                prefill_dtype=layer_inputs.current_kv.dtype,
            )
        if indices.ndim == 2:
            indices = indices.unsqueeze(1)
        flash_mla_sparse_fwd(
            q=q, kv=kv, indices=indices, sm_scale=layer_config.softmax_scale, d_v=layer_config.kv_lora_rank, out=out
        )
        return out

    @staticmethod
    def get_name() -> str:
        return "flashmla_sparse_mla_prefill"

    @classmethod
    def build(cls, metadata: InferStateInfo, context: None = None) -> FlashMLASparsePrefillAttState:
        return FlashMLASparsePrefillAttState(
            mem_index=metadata.mem_index, has_cached_prefix=(metadata.max_cache_len or 0) > 0
        )


class FlashMLASparseDecodeBackend(
    AttentionBackend[SparseMLALayerConfig, FlashMLASparseDecodeAttState, SparseMLADecodeLayerInputs]
):
    def __init__(self, layer_config: SparseMLALayerConfig) -> None:
        super().__init__(layer_config)
        if layer_config.kv_cache_format not in ("fp8_ds_mla", "fp8_group64_e8m0"):
            raise NotImplementedError("FlashMLA sparse decode supports only the packed FP8 caches")

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

        layer_config = self.layer_config
        _check_inputs(layer_config, layer_inputs.q_nope)
        indices = layer_inputs.topk_mem_indices
        if indices.ndim == 2:
            indices = indices.unsqueeze(1)
        assert indices.shape[1] == 1, "FlashMLA sparse decode expects one query row per request"
        packed = layer_inputs.kv_cache
        heads = layer_inputs.q_nope.shape[1]
        if layer_config.kv_cache_format == "fp8_group64_e8m0":
            from lightllm.common.attention.kernels.sparse_mla import fp8_group64_decode
            from lightllm.common.attention.kernels.sparse_mla.pad_query_heads import pad_query_heads

            q_nope = layer_inputs.q_nope
            if fp8_group64_decode.can_use_fp8_sparse_direct_decode(
                q_nope,
                packed,
                indices,
                layer_config.kv_lora_rank,
                att_state.direct_decode_bound,
                is_cuda_graph_capture=att_state.uses_cuda_graph,
            ):
                fp8_group64_decode.fp8_sparse_direct_decode(
                    q_nope, packed, indices, layer_config.softmax_scale, out=out
                )
                return out
            q = pad_query_heads(q_nope, is_cuda_graph_capture=att_state.uses_cuda_graph)
        else:
            q = torch.cat([layer_inputs.q_nope, layer_inputs.q_rope], dim=-1).unsqueeze(1).contiguous()
        kv = torch.as_strided(
            packed,
            size=(packed.shape[0], 1, 1, packed.shape[-1]),
            stride=(packed.stride(0), packed.shape[-1], packed.shape[-1], packed.stride(-1)),
        )
        # FlashMLA writes out in place unless it needs heads padded to its 64-head tile.
        padded = q.shape[2] != heads
        result, _ = flash_mla.flash_mla_with_kvcache(
            q=q,
            k_cache=kv,
            block_table=None,
            cache_seqlens=None,
            head_dim_v=layer_config.kv_lora_rank,
            tile_scheduler_metadata=att_state.sched_meta,
            softmax_scale=layer_config.softmax_scale,
            causal=False,
            is_fp8_kvcache=True,
            indices=indices,
            out=None if padded else out.unsqueeze(1),
        )
        if padded:
            out.copy_(result[:, 0, :heads])
        return out

    @staticmethod
    def get_name() -> str:
        return "flashmla_sparse_mla_decode"

    build_context_type = FlashMLASparseDecodeBuildContext

    @classmethod
    def build(cls, metadata: InferStateInfo, context: FlashMLASparseDecodeBuildContext) -> FlashMLASparseDecodeAttState:
        """A CUDA graph step keeps the captured scheduler metadata per (microbatch, metadata size)."""
        import flash_mla

        if metadata.uses_cuda_graph:
            key = (metadata.microbatch_index, metadata.batch_size)
            if key not in context._cuda_graph_sched_meta:
                context._cuda_graph_sched_meta[key] = flash_mla.get_mla_metadata()[0]
            sched_meta = context._cuda_graph_sched_meta[key]
        else:
            sched_meta = flash_mla.get_mla_metadata()[0]
        return FlashMLASparseDecodeAttState(
            sched_meta=sched_meta,
            uses_cuda_graph=metadata.uses_cuda_graph,
            direct_decode_bound=context.direct_decode_bound,
        )

    @classmethod
    def create_build_context(
        cls, build_key: Optional[int], ctx: AttentionBuildResources
    ) -> FlashMLASparseDecodeBuildContext:
        return FlashMLASparseDecodeBuildContext(direct_decode_bound=build_key)

    @classmethod
    def build_context_key(cls, layer_config: SparseMLALayerConfig) -> Optional[int]:
        return _direct_decode_bound(layer_config)


def _direct_decode_bound(layer_config: SparseMLALayerConfig) -> Optional[int]:
    # 2048 pooled entries plus at most three kpool remainder entries fit the 2112 rounded direct window.
    if (layer_config.topk, layer_config.kpool) == (2048, 4):
        return layer_config.topk + layer_config.kpool - 1
    return None
