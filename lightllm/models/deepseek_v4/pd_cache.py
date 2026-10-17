"""Packed history and request state use the ordinary KV / att_state PD tasks."""

import math

import torch
import triton

from lightllm.common.state_cache_manager.deepseek4 import DSV4_SWA_PAGE_BYTES
from lightllm.server.router.model_infer.mode_backend.pd.cache_page_io import KVCachePageIO
from .triton_kernel.cpu_cache_io import copy_history_pages


def _state_shapes(config):
    # C4 retains the preceding group plus at most three new rows; C128 retains a partial group.
    return (
        ((config.layer_num, triton.cdiv(config.sliding_window, 128) + 1, DSV4_SWA_PAGE_BYTES), torch.uint8),
        ((config.n_c4, 7, 4 * config.head_dim), torch.float32),
        ((config.n_c4, 7, 4 * config.indexer_head_dim), torch.float32),
        ((config.n_c128, 127, 2 * config.head_dim), torch.float32),
    )


def get_pd_page_nbytes(config, page_size):
    history = page_size // 256 * config.history_block_nbytes
    state = sum(math.prod(shape) * dtype.itemsize for shape, dtype in _state_shapes(config))
    return max(history, state)


def _copy_request_state(manager, page, req_idx, seq_len, *, write):
    """Canonical window / live compressor tails; P and D may use different ring sizes."""
    device = manager.swa_buffer.device
    positions = (
        torch.arange(
            max(0, seq_len - manager.cache_config.sliding_window) // 128, triton.cdiv(seq_len, 128), device=device
        ),
        torch.arange(max(0, seq_len - 4 - seq_len % 4), seq_len, device=device),
        torch.arange(max(0, seq_len - 4 - seq_len % 4), seq_len, device=device),
        torch.arange(seq_len - seq_len % 128, seq_len, device=device),
    )
    buffers = (manager.swa_buffer, manager.c4_state_buffer, manager.c4_indexer_state_buffer, manager.c128_state_buffer)
    rings = (manager.swa_pages_per_req, manager.c4_state_ring, manager.c4_state_ring, manager.c128_state_ring)
    data, offset = page.reshape(-1), 0
    for buffer, pos, ring, (shape, dtype) in zip(buffers, positions, rings, _state_shapes(manager.cache_config)):
        nbytes = math.prod(shape) * dtype.itemsize
        view = data[offset : offset + nbytes].view(dtype).view(shape)
        offset += nbytes
        if buffer is None or not pos.numel():
            continue
        indexes = req_idx * ring + pos % ring
        if buffer is manager.swa_buffer and req_idx == manager.req_to_token_indexs.shape[0] - 1:
            indexes.fill_(buffer.shape[1] - 1)
        if write:
            view[:, : pos.numel()].copy_(buffer.index_select(1, indexes), non_blocking=True)
        else:
            buffer.index_copy_(1, indexes, view[:, : pos.numel()].to(device, non_blocking=True))
    if not write and device != page.device:
        # The transport's completion event is recorded on page.device.
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
        )
        return self.mem.pd_kv_page_size // 256 * source.cache_config.history_block_nbytes

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
                )
