"""FlashMLA sparse attention over a sliding-window cache plus an optional compressed cache (DeepSeek V4)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from lightllm.common.attention.backend import AttentionBackend, GraphMode, TensorAllocator
from . import SWASparseMLALayerConfig, SWASparseMLALayerInputs

if TYPE_CHECKING:
    from lightllm.common.basemodel.infer_struct import InferStateInfo


@dataclass(frozen=True)
class FlashMLASWAAttState:
    """FlashMLA scheduler metadata of the group's compress ratio, filled on the step's first kernel call.

    A step being captured also carries a second one, so warm-up runs do not fill it and its planning
    kernels are recorded in the graph to run again on every replay.
    """

    sched_meta: object
    capture_sched_meta: Optional[object] = None


class FlashMLASWAPrefillBackend(
    AttentionBackend[SWASparseMLALayerConfig, FlashMLASWAAttState, SWASparseMLALayerInputs]
):
    name = "flashmla_swa_sparse_mla_prefill"
    phase = "prefill"

    @classmethod
    def group_key(cls, layer_config: SWASparseMLALayerConfig) -> int:
        # Layers with one compress ratio share the scheduler metadata of their kernel calls.
        return layer_config.compress_ratio

    def build(self, infer_state: InferStateInfo) -> FlashMLASWAAttState:
        """Prepare scheduler objects whose GPU planning runs at the first kernel call."""
        import flash_mla

        will_capture = infer_state.graph_mode is GraphMode.CAPTURE
        return FlashMLASWAAttState(
            sched_meta=flash_mla.get_mla_metadata()[0],
            capture_sched_meta=flash_mla.get_mla_metadata()[0] if will_capture else None,
        )

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
        sched_meta = att_state.sched_meta
        if att_state.capture_sched_meta is not None and torch.cuda.is_current_stream_capturing():
            sched_meta = att_state.capture_sched_meta
        flash_mla.flash_mla_with_kvcache(
            q=layer_inputs.q.unsqueeze(1),
            k_cache=layer_inputs.swa_cache,
            block_table=None,
            cache_seqlens=None,
            head_dim_v=layer_config.head_dim_v,
            tile_scheduler_metadata=sched_meta,
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


class FlashMLASWADecodeBackend(FlashMLASWAPrefillBackend):
    name = "flashmla_swa_sparse_mla_decode"
    phase = "decode"
