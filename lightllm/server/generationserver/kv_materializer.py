"""Consumer-side materialization of producer-owned paged KV.

This is deliberately a transitional data plane.  It pays one gather per
downstream task in exchange for a very small consumer contract.  A future
zero-copy provider can replace this module without changing VLA/media server
protocols or model attention code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from lightllm.server.activation_transfer import KVActivation


@dataclass(frozen=True)
class DensePrefixKVLayer:
    """One transformer layer of packed, consumer-owned Prefix K/V."""

    key: torch.Tensor
    value: torch.Tensor

    def __post_init__(self) -> None:
        if self.key.shape != self.value.shape:
            raise ValueError("dense Prefix K/V shapes differ")
        if self.key.ndim != 3:
            raise ValueError("dense Prefix K/V must have shape [token, head, dim]")
        if self.key.device != self.value.device:
            raise ValueError("dense Prefix K/V must be colocated")
        if self.key.dtype != self.value.dtype:
            raise TypeError("dense Prefix K/V dtypes differ")


@dataclass(frozen=True)
class DensePrefixKV:
    """Packed dense Prefix KV for one or more logical requests.

    ``page_table`` is local to this dense snapshot.  It contains only indexes
    into the packed tensors below, never indexes into the producer's KV pool.
    Keeping this tiny layout descriptor lets the same attention kernel support
    ragged batches while fully removing producer storage from model code.
    """

    layers: tuple[DensePrefixKVLayer, ...]
    page_table: torch.Tensor
    prefix_lengths: torch.Tensor
    position_offsets: torch.Tensor
    owners: tuple[tuple[int, int], ...] = ()
    position_offset_values: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.layers:
            raise ValueError("dense Prefix KV requires at least one layer")
        if self.page_table.ndim != 2 or self.page_table.dtype != torch.int32:
            raise TypeError("dense Prefix page table must be a 2-D int32 tensor")
        batch_size = self.page_table.shape[0]
        if self.prefix_lengths.shape != (batch_size,):
            raise ValueError("dense Prefix lengths do not match its batch")
        if self.position_offsets.shape != (batch_size,):
            raise ValueError("dense Prefix position offsets do not match its batch")
        if self.prefix_lengths.dtype != torch.int32:
            raise TypeError("dense Prefix lengths must be int32")
        if self.position_offsets.dtype != torch.int32:
            raise TypeError("dense Prefix position offsets must be int32")
        device = self.layers[0].key.device
        if any(layer.key.device != device for layer in self.layers):
            raise ValueError("all dense Prefix KV layers must be colocated")
        for metadata in (
            self.page_table,
            self.prefix_lengths,
            self.position_offsets,
        ):
            if metadata.device != device:
                raise ValueError("dense Prefix metadata and K/V must be colocated")
        if self.owners and len(self.owners) != batch_size:
            raise ValueError("dense Prefix owners do not match its batch")
        if self.position_offset_values and len(self.position_offset_values) != batch_size:
            raise ValueError("dense Prefix position values do not match its batch")

    @property
    def batch_size(self) -> int:
        return int(self.page_table.shape[0])

    @property
    def layer_count(self) -> int:
        return len(self.layers)

    @property
    def seq_len(self) -> int:
        if self.batch_size != 1:
            raise ValueError("seq_len is only defined for a single dense prefix")
        # The value is supplied by a validated handle, so reading the small
        # CUDA tensor back on the host is unnecessary.
        return int(self.page_table.shape[1])

    @property
    def position_offset(self) -> int:
        if self.batch_size != 1:
            raise ValueError(
                "position_offset is only defined for a single dense prefix"
            )
        if self.position_offset_values:
            return self.position_offset_values[0]
        return int(self.position_offsets[0].item())

    @property
    def layout_key(self) -> tuple:
        """Shape-only CUDA graph key; values are copied before replay."""

        first = self.layers[0].key
        return (
            self.batch_size,
            tuple(int(v) for v in self.page_table.shape),
            tuple(int(v) for v in first.shape),
            len(self.layers),
            str(first.dtype),
        )

    def layer(self, layer_index: int) -> DensePrefixKVLayer:
        return self.layers[layer_index]

    def copy_from(self, source: "DensePrefixKV") -> None:
        """Update graph-owned dense buffers from an identical layout."""

        if self.layout_key != source.layout_key:
            raise ValueError("cannot copy dense Prefix KV between different layouts")
        self.page_table.copy_(source.page_table, non_blocking=True)
        self.prefix_lengths.copy_(source.prefix_lengths, non_blocking=True)
        self.position_offsets.copy_(source.position_offsets, non_blocking=True)
        for target_layer, source_layer in zip(
            self.layers, source.layers, strict=True
        ):
            target_layer.key.copy_(source_layer.key, non_blocking=True)
            target_layer.value.copy_(source_layer.value, non_blocking=True)


def _as_handles(
    handles: KVActivation | Sequence[KVActivation],
) -> tuple[KVActivation, ...]:
    if isinstance(handles, KVActivation):
        return (handles,)
    result = tuple(handles)
    if not result or not all(isinstance(handle, KVActivation) for handle in result):
        raise TypeError("gather requires one or more KVActivation objects")
    return result


@torch.no_grad()
def gather_dense_prefix_kv(
    mem_manager,
    req_to_token_indexs: torch.Tensor,
    handles: KVActivation | Sequence[KVActivation],
) -> DensePrefixKV:
    """Gather rank-local paged KV into one packed, consumer-owned snapshot."""

    handles = _as_handles(handles)
    if req_to_token_indexs.ndim != 2:
        raise ValueError("request-to-token table must be two-dimensional")
    if req_to_token_indexs.dtype != torch.int32:
        raise TypeError("request-to-token table must be int32")
    row_count, row_width = req_to_token_indexs.shape
    for handle in handles:
        if handle.infer_req_idx >= row_count:
            raise ValueError("Prefix KV request row is outside the shared table")
        if handle.seq_len > row_width:
            raise ValueError("Prefix KV length exceeds the shared request row")

    device = req_to_token_indexs.device
    lengths = torch.tensor(
        [handle.seq_len for handle in handles],
        dtype=torch.int32,
        device=device,
    )
    offsets = torch.tensor(
        [handle.position_offset for handle in handles],
        dtype=torch.int32,
        device=device,
    )
    max_length = max(handle.seq_len for handle in handles)
    page_table = torch.zeros(
        (len(handles), max_length),
        dtype=torch.int32,
        device=device,
    )
    producer_indexes = []
    packed_offset = 0
    for batch_index, handle in enumerate(handles):
        producer_indexes.append(
            req_to_token_indexs[
                handle.infer_req_idx,
                : handle.seq_len,
            ]
        )
        page_table[batch_index, : handle.seq_len] = torch.arange(
            packed_offset,
            packed_offset + handle.seq_len,
            dtype=torch.int32,
            device=device,
        )
        packed_offset += handle.seq_len
    gather_indexes = torch.cat(producer_indexes).to(dtype=torch.int64)

    layer_count = int(mem_manager.layer_num)
    kv_buffer = getattr(mem_manager, "kv_buffer", None)
    if (
        isinstance(kv_buffer, torch.Tensor)
        and kv_buffer.is_cuda
        and kv_buffer.device == device
        and kv_buffer.ndim == 4
        and kv_buffer.shape[0] == layer_count
        and kv_buffer.shape[2] == 2 * int(mem_manager.head_num)
        and torch.is_floating_point(kv_buffer)
    ):
        # One launch gathers every layer and both K/V planes into a layout
        # whose individual layer views are contiguous. This is the normal
        # production path for unquantized LightLLM KV pools.
        from .triton_kv_gather import triton_gather_dense_prefix_kv

        packed_kv = triton_gather_dense_prefix_kv(
            kv_buffer,
            gather_indexes,
            head_count=int(mem_manager.head_num),
        )
        layers = [
            DensePrefixKVLayer(packed_kv[layer_index, 0], packed_kv[layer_index, 1])
            for layer_index in range(layer_count)
        ]
    else:
        # CPU tests and storage implementations without the ordinary combined
        # KV pool retain the clear per-layer reference path.
        layers = []
        for layer_index in range(layer_count):
            key_pool, value_pool = mem_manager.get_att_input_params(layer_index)
            if not isinstance(key_pool, torch.Tensor) or not isinstance(
                value_pool, torch.Tensor
            ):
                raise TypeError(
                    "dense Prefix gather currently requires unquantized tensor K/V"
                )
            if key_pool.device != device or value_pool.device != device:
                raise ValueError("Prefix KV pool and request table must be colocated")
            key = torch.index_select(key_pool, 0, gather_indexes).contiguous()
            value = torch.index_select(value_pool, 0, gather_indexes).contiguous()
            layers.append(DensePrefixKVLayer(key, value))

    return DensePrefixKV(
        layers=tuple(layers),
        page_table=page_table,
        prefix_lengths=lengths,
        position_offsets=offsets,
        owners=tuple((handle.shm_index, handle.request_id) for handle in handles),
        position_offset_values=tuple(
            int(handle.position_offset) for handle in handles
        ),
    )


class PagedKVMaterializer:
    """Generation-worker adapter over one producer rank's shared KV pool."""

    def __init__(
        self,
        mem_manager,
        req_to_token_indexs: torch.Tensor,
        *,
        device_index: int,
    ) -> None:
        if (
            req_to_token_indexs.device.type != "cuda"
            or req_to_token_indexs.device.index != device_index
        ):
            raise ValueError(
                "generation worker must colocate with its producer KV rank: "
                f"worker=cuda:{device_index}, KV={req_to_token_indexs.device}"
            )
        self.mem_manager = mem_manager
        self.req_to_token_indexs = req_to_token_indexs

    @classmethod
    def from_shared(
        cls,
        rank_in_node: int,
        *,
        device_index: int,
    ) -> "PagedKVMaterializer":
        from lightllm.common.kv_cache_mem_manager import MemoryManager
        from lightllm.common.req_to_token_ipc import (
            load_req_to_token_indexs_from_shm,
        )

        return cls(
            MemoryManager.loads_from_shm(rank_in_node),
            load_req_to_token_indexs_from_shm(rank_in_node),
            device_index=device_index,
        )

    @property
    def prefix_layout(self) -> tuple[int, int, int]:
        return (
            int(self.mem_manager.layer_num),
            int(self.mem_manager.head_num),
            int(self.mem_manager.head_dim),
        )

    @property
    def max_seq_length(self) -> int:
        return int(self.req_to_token_indexs.shape[1])

    def gather(
        self,
        activations: KVActivation | Sequence[KVActivation],
    ) -> DensePrefixKV:
        return gather_dense_prefix_kv(
            self.mem_manager,
            self.req_to_token_indexs,
            activations,
        )


__all__ = [
    "DensePrefixKV",
    "DensePrefixKVLayer",
    "PagedKVMaterializer",
    "gather_dense_prefix_kv",
]
