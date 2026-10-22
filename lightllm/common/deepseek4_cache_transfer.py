"""Packed history and request state use the ordinary KV / att_state PD tasks."""

import math
from dataclasses import dataclass

import torch

from lightllm.server.router.model_infer.mode_backend.pd.cache_page_io import KVCachePageIO
from lightllm.models.deepseek_v4.triton_kernel.cache_io import copy_history_pages
from lightllm.models.deepseek_v4.triton_kernel.swa_cache import gather_swa_tokens, scatter_swa_tokens


def _state_shapes(config):
    short, long = config.short_compress_ratio, config.long_compress_ratio
    # The short compressor overlaps its preceding group; the long one does not.
    return (
        ((config.layer_num, config.sliding_window, config.kv_bytes), torch.uint8),
        ((config.n_c4, 2 * short - 1, 4 * config.head_dim), torch.float32),
        ((config.n_c4, 2 * short - 1, 4 * config.indexer_head_dim), torch.float32),
        ((config.n_c128, long - 1, 2 * config.head_dim), torch.float32),
    )


def get_pd_page_nbytes(config, page_size):
    history = page_size * sum(config.history_widths)
    state = sum(math.prod(shape) * dtype.itemsize for shape, dtype in _state_shapes(config))
    return max(history, state)


def _copy_request_state(manager, page, req_idx, seq_len, *, write):
    """Transfer chronological tokens; runtime ring capacities may differ on P and D."""
    config = manager.cache_config
    device = manager.swa_buffer.device
    short, long = config.short_compress_ratio, config.long_compress_ratio
    positions = (
        torch.arange(seq_len - config.sliding_window, seq_len, device=device),
        torch.arange(max(0, seq_len - short - seq_len % short), seq_len, device=device),
        torch.arange(max(0, seq_len - short - seq_len % short), seq_len, device=device),
        torch.arange(seq_len - seq_len % long, seq_len, device=device),
    )
    buffers = (None, manager.req_to_c4_state, manager.req_to_c4_indexer_state, manager.req_to_c128_state)
    data, offset = page.reshape(-1), 0
    for component, (buffer, pos, (shape, dtype)) in enumerate(zip(buffers, positions, _state_shapes(config))):
        nbytes = math.prod(shape) * dtype.itemsize
        view = data[offset : offset + nbytes].view(dtype).view(shape)
        offset += nbytes
        if component == 0:
            if write:
                view.copy_(gather_swa_tokens(manager, req_idx, pos, prefill=False), non_blocking=True)
            else:
                scatter_swa_tokens(manager, req_idx, pos, view.to(device, non_blocking=True))
        elif buffer is not None and pos.numel():
            slots = pos % buffer.shape[2]
            if write:
                view[:, : pos.numel()].copy_(buffer[:, req_idx].index_select(1, slots), non_blocking=True)
            else:
                buffer[:, req_idx].index_copy_(1, slots, view[:, : pos.numel()].to(device, non_blocking=True))
    if not write and device != page.device:
        torch.cuda.current_stream(device).synchronize()
    return offset


@dataclass(frozen=True)
class DeepseekV4KVTransferView:
    kv_buffer: tuple[torch.Tensor, ...]
    cache_config: object

    def create_page_io(self, kv_views, state_transfers, device_id, dp_world_size):
        return DeepseekV4CachePageIO(kv_views, state_transfers, device_id, dp_world_size)


@dataclass(frozen=True)
class DeepseekV4StateTransferView:
    cache_config: object
    swa_buffer: torch.Tensor
    swa_capacity_tokens: int
    HOLD_SWA_SLOT: int
    HOLD_REQUEST_ID: int
    req_to_c4_state: object
    req_to_c4_indexer_state: object
    req_to_c128_state: object

    @classmethod
    def from_state_manager(cls, state):
        return cls(**{name: getattr(state, name) for name in cls.__dataclass_fields__})


class DeepseekV4CachePageIO(KVCachePageIO):
    def __init__(self, kv_views, state_transfers, device_id, dp_world_size):
        self.kv_views, self.state_transfers = kv_views, state_transfers
        self.device_id, self.dp_world_size = device_id, dp_world_size

    def alloc_buffer(self, page_num, page_size):
        self.page_size = page_size
        config = self.kv_views[self.device_id].cache_config
        self.buffer = torch.empty((page_num, get_pd_page_nbytes(config, page_size)), dtype=torch.uint8, device="cuda")
        self.index_buffers = [torch.empty(page_size, dtype=torch.int64, pin_memory=True) for _ in range(page_num)]
        return self.buffer

    def _indexes(self, task, page_index):
        indexes = self.index_buffers[page_index][: len(task.mem_indexes)]
        indexes.numpy()[:] = task.mem_indexes
        return indexes.cuda(non_blocking=True)

    def pack(self, task):
        source = self.kv_views[task.prefill_dp_index * self.dp_world_size]
        page = self.buffer[task.src_page_index]
        if task.page_kind == "att_state":
            return _copy_request_state(
                self.state_transfers[task.prefill_dp_index * self.dp_world_size],
                page,
                task.req_idx,
                task.start_kv_index,
                write=True,
            )
        assert task.page_kind == "kv"
        indexes = self._indexes(task, task.src_page_index)
        copy_history_pages(
            source,
            indexes,
            torch.zeros(1, dtype=torch.int32, device=page.device),
            page.view(1, -1),
            to_pages=True,
            page_tokens=self.page_size,
            round_tail=True,
        )
        return self.page_size * sum(source.cache_config.history_widths)

    def unpack(self, task):
        page = self.buffer[task.dst_page_index]
        indexes = self._indexes(task, task.dst_page_index) if task.page_kind == "kv" else None
        first = task.decode_dp_index * self.dp_world_size
        for rank in range(first, first + self.dp_world_size):
            destination = self.kv_views[rank]
            if task.page_kind == "att_state":
                _copy_request_state(self.state_transfers[rank], page, task.req_idx, task.start_kv_index, write=False)
            else:
                assert task.page_kind == "kv"
                copy_history_pages(
                    destination,
                    indexes,
                    torch.zeros(1, dtype=torch.int32, device=page.device),
                    page.view(1, -1),
                    to_pages=False,
                    page_tokens=self.page_size,
                    round_tail=True,
                )


class DeepseekV4CpuCacheTransfer:
    def __init__(self, mem_manager, big_page_buffers):
        self.kv_buffer = mem_manager.kv_buffer
        self.cache_config = mem_manager.cache_config
        self.big_page_buffers = big_page_buffers

    def load(self, mem_indexes, page_indexes, cpu_cache_client, slots):
        from lightllm.models.deepseek_v4.triton_kernel.cache_io import copy_checkpoint_pages

        pages = cpu_cache_client.cpu_kv_cache_tensor.view(cpu_cache_client.cpu_kv_cache_tensor.shape[0], -1)
        copy_history_pages(self, mem_indexes, page_indexes, pages, to_pages=False)
        ids = torch.tensor(slots, dtype=torch.int64, device="cuda")
        copy_checkpoint_pages(self, ids, page_indexes, pages, to_pages=False)

    def offload(self, mem_indexes, page_indexes, page_readies, cpu_cache_client, slots, tail_state):
        from lightllm.utils.dist_utils import get_current_rank_in_dp
        from lightllm.utils.custom_kernel_utis import tensor_copy_3dim
        from lightllm.models.deepseek_v4.triton_kernel.cache_io import copy_checkpoint_pages

        if get_current_rank_in_dp() != 0:
            return
        if tail_state is not None:
            for dst, src in zip(self.big_page_buffers.get_state_cache(slots[-1]), tail_state):
                if src.numel():
                    tensor_copy_3dim(dst, src)
        pages = cpu_cache_client.cpu_kv_cache_tensor.view(cpu_cache_client.cpu_kv_cache_tensor.shape[0], -1)
        ids = torch.tensor(slots, dtype=torch.int64, device="cuda")
        copy_history_pages(self, mem_indexes, page_indexes, pages, to_pages=True, readies=page_readies)
        copy_checkpoint_pages(self, ids, page_indexes, pages, to_pages=True, readies=page_readies)
