"""Route recording bound to native storage before batch and Graph preparation."""

from dataclasses import dataclass
from functools import partial

import torch

from lightllm.common.basemodel.triton_kernel.routing_capture import scatter_routing_topk_to_cpu


@dataclass(frozen=True)
class RouteCapture:
    buffer: torch.Tensor
    pointer: torch.Tensor
    layer_slots: tuple[tuple[int, int], ...]
    dtype_id: int

    def bind_batch(self, mem_indices):
        if not mem_indices.is_cuda or not mem_indices.is_contiguous():
            raise ValueError("route recording requires the batch's contiguous CUDA token indices")
        return {layer: partial(self.capture, slot, mem_indices) for layer, slot in self.layer_slots}

    def capture(self, slot, mem_indices, topk_ids):
        capacity, layers, topk = self.buffer.shape
        if topk_ids.ndim != 2 or topk_ids.shape[1] != topk or len(mem_indices) < len(topk_ids):
            raise ValueError("route recording disagrees with the bound token or expert layout")
        scatter_routing_topk_to_cpu(
            topk_ids=topk_ids.contiguous(),
            mem_indexes=mem_indices,
            routing_buffer_ptr=self.pointer,
            moe_layer_index=slot,
            num_moe_layers=layers,
            kv_cache_size=capacity,
            topk=topk,
            dtype_id=self.dtype_id,
        )
