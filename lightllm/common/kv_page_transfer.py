import torch

from lightllm.common.kv_trans_kernel.nixl_kv_trans import page_io, mla_page_io


class KVPageTransfer:
    def __init__(self, view):
        if view.page_shape is None:
            raise ValueError("PD transfer requires a KV page layout")
        self.view = view

    def alloc_paged_kv_move_buffer(self, page_num, page_size):
        self.kv_move_buffer = torch.empty(
            (page_num, page_size, *self.view.page_shape), dtype=self.view.page_dtype, device="cuda"
        )
        self.index_buffers = [torch.empty(page_size, dtype=torch.int64, pin_memory=True) for _ in range(page_num)]
        return self.kv_move_buffer

    def write_mem_to_page_kv_move_buffer(self, mem_indexes, page_index, dp_index, kv_views, dp_world_size):
        self._copy(mem_indexes, page_index, dp_index, kv_views, dp_world_size, "write")

    def read_page_kv_move_buffer_to_mem(self, mem_indexes, page_index, dp_index, kv_views, dp_world_size):
        self._copy(mem_indexes, page_index, dp_index, kv_views, dp_world_size, "read")

    def _copy(self, indexes, page_index, dp_index, views, dp_world_size, mode):
        pinned = self.index_buffers[page_index][: len(indexes)]
        pinned.numpy()[:] = indexes
        indexes_gpu = pinned.cuda(non_blocking=True)
        page = self.kv_move_buffer[page_index]
        ranks = views[dp_index * dp_world_size : (dp_index + 1) * dp_world_size]
        assert len(ranks) == dp_world_size
        if self.view.replicated:
            for view in ranks[:1] if mode == "write" else ranks:
                mla_page_io(mem_indexes=indexes_gpu, page_tensor=page, kv_buffer=view.kv_buffer, mode=mode)
        else:
            repeat_count = dp_world_size * self.view.kv_buffer.shape[2] // self.kv_move_buffer.shape[3]
            for rank, view in enumerate(ranks):
                if mode == "read" or rank % repeat_count == 0:
                    page_io(
                        mem_indexes=indexes_gpu,
                        page_tensor=page,
                        kv_buffer=view.kv_buffer,
                        tp_index=rank,
                        tp_world_size=dp_world_size,
                        mode=mode,
                    )
