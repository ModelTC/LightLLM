"""Packed history and request state use the ordinary KV / att_state PD tasks."""

import math

import torch

from lightllm.server.router.model_infer.mode_backend.pd.cache_page_io import KVCachePageIO
from .triton_kernel.cache_io import copy_history_pages
from .triton_kernel.swa_cache import gather_swa_tokens, scatter_swa_tokens


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


class DeepseekV4CachePageIO(KVCachePageIO):
    def _indexes(self, task, page_index):
        indexes = self.mem._buffer_mem_indexes_tensors[page_index][: len(task.mem_indexes)]
        indexes.numpy()[:] = task.mem_indexes
        return indexes.cuda(non_blocking=True)

    def pack(self, task):
        source = self.mem_managers[task.prefill_dp_index * self.dp_world_size]
        page = self.mem.kv_move_buffer[task.src_page_index]
        if task.page_kind == "att_state":
            return _copy_request_state(source, page, task.req_idx, task.start_kv_index, write=True)
        assert task.page_kind == "kv"
        indexes = self._indexes(task, task.src_page_index)
        copy_history_pages(
            source,
            indexes,
            torch.zeros(1, dtype=torch.int32, device=page.device),
            page.view(1, -1),
            to_pages=True,
            page_tokens=self.mem.pd_kv_page_size,
            round_tail=True,
        )
        return self.mem.pd_kv_page_size * sum(source.cache_config.history_widths)

    def unpack(self, task):
        page = self.mem.kv_move_buffer[task.dst_page_index]
        indexes = self._indexes(task, task.dst_page_index) if task.page_kind == "kv" else None
        first = task.decode_dp_index * self.dp_world_size
        for destination in self.mem_managers[first : first + self.dp_world_size]:
            if task.page_kind == "att_state":
                _copy_request_state(destination, page, task.req_idx, task.start_kv_index, write=False)
            else:
                assert task.page_kind == "kv"
                copy_history_pages(
                    destination,
                    indexes,
                    torch.zeros(1, dtype=torch.int32, device=page.device),
                    page.view(1, -1),
                    to_pages=False,
                    page_tokens=self.mem.pd_kv_page_size,
                    round_tail=True,
                )
