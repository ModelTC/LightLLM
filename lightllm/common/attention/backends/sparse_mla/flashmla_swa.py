"""FlashMLA sparse attention over a sliding-window cache plus an optional compressed cache (DeepSeek V4)."""

from dataclasses import dataclass
from typing import Optional

import torch

from lightllm.common.attention.backend import (
    AttentionBackend,
    AttentionBuildResources,
    TensorAllocator,
)
from lightllm.common.basemodel.infer_struct import GraphMode, InferStateInfo
from . import SWASparseMLALayerConfig, SWASparseMLALayerInputs


@dataclass(frozen=True)
class FlashMLASWAAttState:
    """FlashMLA scheduler metadata per compress ratio, filled on the step's first kernel call per ratio.

    A step being captured also carries a second set, so warm-up runs do not
    fill it and its planning kernels are recorded in the graph to run again on every replay.
    """

    sched_meta_by_ratio: dict
    capture_sched_meta_by_ratio: Optional[dict] = None


class FlashMLASWABuildContext:
    def __init__(self, *, compress_ratios: tuple) -> None:
        self.compress_ratios = compress_ratios


class _FlashMLASWABackend(AttentionBackend[SWASparseMLALayerConfig, FlashMLASWAAttState, SWASparseMLALayerInputs]):
    def forward(
        self,
        att_state: FlashMLASWAAttState,
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
        sched_meta_by_ratio = att_state.sched_meta_by_ratio
        if att_state.capture_sched_meta_by_ratio is not None and torch.cuda.is_current_stream_capturing():
            sched_meta_by_ratio = att_state.capture_sched_meta_by_ratio
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

    build_context_type = FlashMLASWABuildContext

    @classmethod
    def build(cls, metadata: Optional[InferStateInfo], context: FlashMLASWABuildContext) -> FlashMLASWAAttState:
        """Prepare scheduler objects whose GPU planning runs at the first kernel call."""
        import flash_mla

        def new_set():
            return {ratio: flash_mla.get_mla_metadata()[0] for ratio in context.compress_ratios}

        will_capture = metadata is not None and metadata.graph_mode is GraphMode.CAPTURE
        return FlashMLASWAAttState(
            sched_meta_by_ratio=new_set(),
            capture_sched_meta_by_ratio=new_set() if will_capture else None,
        )

    @classmethod
    def create_build_context(cls, build_key: int, ctx: AttentionBuildResources) -> FlashMLASWABuildContext:
        # Layers with one compress ratio share the scheduler metadata of their kernel calls.
        return FlashMLASWABuildContext(compress_ratios=(build_key,))

    @classmethod
    def build_context_key(cls, layer_config: SWASparseMLALayerConfig) -> int:
        return layer_config.compress_ratio


class FlashMLASWAPrefillBackend(_FlashMLASWABackend):
    @staticmethod
    def get_name() -> str:
        return "flashmla_swa_sparse_mla_prefill"


class FlashMLASWADecodeBackend(_FlashMLASWABackend):
    @staticmethod
    def get_name() -> str:
        return "flashmla_swa_sparse_mla_decode"
