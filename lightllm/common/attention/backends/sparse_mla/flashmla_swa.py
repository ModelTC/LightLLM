"""FlashMLA sparse attention over a sliding-window cache plus an optional compressed cache (DeepSeek V4)."""

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
from . import SWASparseMLALayerConfig, SWASparseMLALayerInputs


@dataclass(frozen=True)
class FlashMLASWAStepArgs:
    """FlashMLA scheduler metadata per compress ratio, filled on the step's first kernel call per ratio.

    A decode step also carries a second set used while a CUDA graph is captured, so warm-up runs do not
    fill it and its planning kernels are recorded in the graph to run again on every replay.
    """

    sched_meta_by_ratio: dict
    capture_sched_meta_by_ratio: Optional[dict] = None


class FlashMLASWAStepBuilder(AttentionStepBuilder[FlashMLASWAStepArgs]):
    def __init__(self, *, compress_ratios: tuple, for_cuda_graph_capture: bool) -> None:
        """compress_ratios lists the model's distinct ratios; for_cuda_graph_capture adds the capture set (decode)."""
        self.compress_ratios = compress_ratios
        self.for_cuda_graph_capture = for_cuda_graph_capture

    def build(self, batch: Optional[AttentionBatch] = None) -> FlashMLASWAStepArgs:
        """The metadata is planned on the GPU at the first kernel call, so it does not depend on batch."""
        import flash_mla

        def new_set():
            return {ratio: flash_mla.get_mla_metadata()[0] for ratio in self.compress_ratios}

        return FlashMLASWAStepArgs(
            sched_meta_by_ratio=new_set(),
            capture_sched_meta_by_ratio=new_set() if self.for_cuda_graph_capture else None,
        )


class FlashMLASWAImpl(AttentionImpl[SWASparseMLALayerConfig, FlashMLASWAStepArgs, SWASparseMLALayerInputs]):
    def forward(
        self,
        step_args: FlashMLASWAStepArgs,
        layer_inputs: SWASparseMLALayerInputs,
        *,
        out: torch.Tensor,
        alloc_func: TensorAllocator = torch.empty,
    ) -> torch.Tensor:
        """Fill and return out [rows, num_heads, head_dim_v], written by FlashMLA in place."""
        import flash_mla

        layer_config = self.layer_config
        assert layer_inputs.q.shape[1] == layer_config.num_heads, "q does not match the layer_config"
        assert (layer_inputs.compressed_cache is not None) == (layer_config.compress_ratio != 0)
        assert (layer_inputs.attn_sink is not None) == layer_config.has_attn_sink
        sched_meta_by_ratio = step_args.sched_meta_by_ratio
        if step_args.capture_sched_meta_by_ratio is not None and torch.cuda.is_current_stream_capturing():
            sched_meta_by_ratio = step_args.capture_sched_meta_by_ratio
        flash_mla.flash_mla_with_kvcache(
            q=layer_inputs.q.unsqueeze(1),
            k_cache=layer_inputs.swa_cache,
            block_table=None,
            cache_seqlens=None,
            head_dim_v=layer_config.head_dim_v,
            tile_scheduler_metadata=sched_meta_by_ratio[layer_config.compress_ratio],
            num_splits=None,
            softmax_scale=layer_config.softmax_scale,
            causal=False,
            is_fp8_kvcache=True,
            indices=layer_inputs.swa_indices,
            attn_sink=layer_inputs.attn_sink,
            topk_length=layer_inputs.swa_lengths,
            extra_k_cache=layer_inputs.compressed_cache,
            extra_indices_in_kvcache=layer_inputs.extra_indices,
            extra_topk_length=layer_inputs.extra_lengths,
            out=out.unsqueeze(1),
        )
        return out


class FlashMLASWAPrefillBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "flashmla_swa_sparse_mla_prefill"

    @staticmethod
    def get_step_builder_cls() -> type[FlashMLASWAStepBuilder]:
        return FlashMLASWAStepBuilder

    @staticmethod
    def get_impl_cls() -> type[FlashMLASWAImpl]:
        return FlashMLASWAImpl

    @classmethod
    def create_step_builder(
        cls, layer_config: SWASparseMLALayerConfig, ctx: StepBuilderContext
    ) -> FlashMLASWAStepBuilder:
        # Layers with one compress ratio share the scheduler metadata of their kernel calls.
        return FlashMLASWAStepBuilder(compress_ratios=(layer_config.compress_ratio,), for_cuda_graph_capture=False)

    @classmethod
    def step_builder_key(cls, layer_config: SWASparseMLALayerConfig) -> int:
        return layer_config.compress_ratio


class FlashMLASWADecodeBackend(FlashMLASWAPrefillBackend):
    @staticmethod
    def get_name() -> str:
        return "flashmla_swa_sparse_mla_decode"

    @classmethod
    def create_step_builder(
        cls, layer_config: SWASparseMLALayerConfig, ctx: StepBuilderContext
    ) -> FlashMLASWAStepBuilder:
        return FlashMLASWAStepBuilder(compress_ratios=(layer_config.compress_ratio,), for_cuda_graph_capture=True)
