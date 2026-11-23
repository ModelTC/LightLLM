"""FlashMLA sparse MLA: prefill over bf16 or packed FP8 caches and decode over packed FP8 caches."""

from dataclasses import dataclass
from typing import Optional

import torch

from lightllm.common.attention.backend import (
    AttentionBackend,
    AttentionBatch,
    AttentionImpl,
    AttentionStepBuilder,
    StepBuilderContext,
    TensorAllocator,
)
from . import SparseMLADecodeLayerInputs, SparseMLALayerConfig, SparseMLAPrefillLayerInputs


@dataclass(frozen=True)
class FlashMLASparsePrefillStepArgs:
    """mem_index holds the cache slots of the step's new tokens; has_cached_prefix says whether any
    request reads keys from earlier steps."""

    mem_index: Optional[torch.Tensor]
    has_cached_prefix: bool


@dataclass(frozen=True)
class FlashMLASparseDecodeStepArgs:
    """FlashMLA's tile scheduler metadata, filled on the step's first kernel call.

    direct_decode_bound, when set, bounds the valid entries of every top-k row, which lets small
    CUDA graph steps over the "fp8_group64_e8m0" cache use a direct Triton kernel.
    """

    sched_meta: object
    uses_cuda_graph: bool = False
    direct_decode_bound: Optional[int] = None


class FlashMLASparsePrefillStepBuilder(AttentionStepBuilder[FlashMLASparsePrefillStepArgs]):
    def build(self, batch: AttentionBatch) -> FlashMLASparsePrefillStepArgs:
        return FlashMLASparsePrefillStepArgs(mem_index=batch.mem_index, has_cached_prefix=batch.max_ready_cache_len > 0)


class FlashMLASparseDecodeStepBuilder(AttentionStepBuilder[FlashMLASparseDecodeStepArgs]):
    def __init__(self, *, direct_decode_bound: Optional[int] = None) -> None:
        self.direct_decode_bound = direct_decode_bound
        self._cuda_graph_sched_meta: dict[tuple[int, int], object] = {}

    def build(self, batch: AttentionBatch) -> FlashMLASparseDecodeStepArgs:
        """A CUDA graph step keeps the captured scheduler metadata per (microbatch, batch size)."""
        import flash_mla

        if batch.uses_cuda_graph:
            key = (batch.microbatch_index, batch.batch_size)
            if key not in self._cuda_graph_sched_meta:
                self._cuda_graph_sched_meta[key] = flash_mla.get_mla_metadata()[0]
            sched_meta = self._cuda_graph_sched_meta[key]
        else:
            sched_meta = flash_mla.get_mla_metadata()[0]
        return FlashMLASparseDecodeStepArgs(
            sched_meta=sched_meta,
            uses_cuda_graph=batch.uses_cuda_graph,
            direct_decode_bound=self.direct_decode_bound,
        )


def _check_inputs(layer_config: SparseMLALayerConfig, q: torch.Tensor) -> None:
    assert q.shape[1] == layer_config.num_heads and q.dtype == layer_config.dtype, "q does not match the layer_config"


class FlashMLASparsePrefillImpl(
    AttentionImpl[SparseMLALayerConfig, FlashMLASparsePrefillStepArgs, SparseMLAPrefillLayerInputs]
):
    def __init__(self, layer_config: SparseMLALayerConfig) -> None:
        super().__init__(layer_config)
        if layer_config.kv_cache_format not in ("bf16", "fp8_ds_mla"):
            raise NotImplementedError(f"FlashMLA sparse prefill does not support {layer_config.kv_cache_format!r}")

    def forward(
        self,
        step_args: FlashMLASparsePrefillStepArgs,
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
        heads = q.shape[1]
        if layer_config.kv_cache_format == "bf16":
            from sgl_kernel.flash_mla import flash_mla_sparse_fwd

            kv, indices = layer_inputs.kv_cache, layer_inputs.topk_mem_indices
            # FlashMLA uses 64-head tiles; pad other head counts with zero queries.
            padded_heads = -(-heads // 64) * 64
            if padded_heads != heads:
                padded = q.new_zeros((q.shape[0], padded_heads, q.shape[2]))
                padded[:, :heads] = q
                q = padded
        else:
            from flash_mla import flash_mla_sparse_fwd

            assert layer_inputs.current_kv is not None and layer_inputs.topk_indices is not None
            kv, indices = layer_inputs.current_kv, layer_inputs.topk_indices
            if step_args.has_cached_prefix:
                from lightllm.common.basemodel.triton_kernel.kv_copy.prefill_compact_kv_flashmla_fp8 import (
                    get_prefill_kv_cache_and_remap_indices_triton,
                )

                kv, indices = get_prefill_kv_cache_and_remap_indices_triton(
                    packed_kv=layer_inputs.kv_cache,
                    topk_mem_indices=layer_inputs.topk_mem_indices,
                    prefill_mem_index=step_args.mem_index,
                    prefill_cache_kv=layer_inputs.current_kv,
                    prefill_dtype=layer_inputs.current_kv.dtype,
                )
        if indices.ndim == 2:
            indices = indices.unsqueeze(1)
        result, _, _ = flash_mla_sparse_fwd(
            q=q, kv=kv, indices=indices, sm_scale=layer_config.softmax_scale, d_v=layer_config.kv_lora_rank
        )
        # FlashMLA allocates its own output.
        out.copy_(result[:, :heads])
        return out


class FlashMLASparseDecodeImpl(
    AttentionImpl[SparseMLALayerConfig, FlashMLASparseDecodeStepArgs, SparseMLADecodeLayerInputs]
):
    def __init__(self, layer_config: SparseMLALayerConfig) -> None:
        super().__init__(layer_config)
        if layer_config.kv_cache_format not in ("fp8_ds_mla", "fp8_group64_e8m0"):
            raise NotImplementedError("FlashMLA sparse decode supports only the packed FP8 caches")

    def forward(
        self,
        step_args: FlashMLASparseDecodeStepArgs,
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
                step_args.direct_decode_bound,
                is_cuda_graph_capture=step_args.uses_cuda_graph,
            ):
                out.copy_(
                    fp8_group64_decode.fp8_sparse_direct_decode(q_nope, packed, indices, layer_config.softmax_scale)
                )
                return out
            q = pad_query_heads(q_nope, is_cuda_graph_capture=step_args.uses_cuda_graph)
        else:
            q = torch.cat([layer_inputs.q_nope, layer_inputs.q_rope], dim=-1).unsqueeze(1).contiguous()
        kv = torch.as_strided(
            packed,
            size=(packed.shape[0], 1, 1, packed.shape[-1]),
            stride=(packed.stride(0), packed.shape[-1], packed.shape[-1], packed.stride(-1)),
        )
        result, _ = flash_mla.flash_mla_with_kvcache(
            q=q,
            k_cache=kv,
            block_table=None,
            cache_seqlens=None,
            head_dim_v=layer_config.kv_lora_rank,
            tile_scheduler_metadata=step_args.sched_meta,
            softmax_scale=layer_config.softmax_scale,
            causal=False,
            is_fp8_kvcache=True,
            indices=indices,
        )
        # FlashMLA allocates its own output.
        out.copy_(result[:, 0, :heads])
        return out


class FlashMLASparsePrefillBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "flashmla_sparse_mla_prefill"

    @staticmethod
    def get_step_builder_cls() -> type[FlashMLASparsePrefillStepBuilder]:
        return FlashMLASparsePrefillStepBuilder

    @staticmethod
    def get_impl_cls() -> type[FlashMLASparsePrefillImpl]:
        return FlashMLASparsePrefillImpl

    @classmethod
    def create_step_builder(
        cls, layer_config: SparseMLALayerConfig, ctx: StepBuilderContext
    ) -> FlashMLASparsePrefillStepBuilder:
        return FlashMLASparsePrefillStepBuilder()


class FlashMLASparseDecodeBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "flashmla_sparse_mla_decode"

    @staticmethod
    def get_step_builder_cls() -> type[FlashMLASparseDecodeStepBuilder]:
        return FlashMLASparseDecodeStepBuilder

    @staticmethod
    def get_impl_cls() -> type[FlashMLASparseDecodeImpl]:
        return FlashMLASparseDecodeImpl

    @classmethod
    def create_step_builder(
        cls, layer_config: SparseMLALayerConfig, ctx: StepBuilderContext
    ) -> FlashMLASparseDecodeStepBuilder:
        return FlashMLASparseDecodeStepBuilder(direct_decode_bound=_direct_decode_bound(layer_config))

    @classmethod
    def step_builder_key(cls, layer_config: SparseMLALayerConfig) -> Optional[int]:
        return _direct_decode_bound(layer_config)


def _direct_decode_bound(layer_config: SparseMLALayerConfig) -> Optional[int]:
    # 2048 pooled entries plus at most three kpool remainder entries fit the 2112 rounded direct window.
    if (layer_config.topk, layer_config.kpool) == (2048, 4):
        return layer_config.topk + layer_config.kpool - 1
    return None
