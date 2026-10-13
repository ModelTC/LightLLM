import torch

from .base import StateCacheManager
from .layer_cache import LayerCache


class DeepseekV4StateCacheManager(StateCacheManager):
    """Pinned continuation checkpoints; compressed history stays in token pages."""

    def __init__(self, size, layout, keep_num=0, shm_name=None, shm_create=False):
        super().__init__(size, keep_num, shm_name, shm_create)
        self.layout = layout
        state_nbytes = layout.page_nbytes - layout.swa_offset
        if shm_name is None:
            self.buffer = torch.empty((size, state_nbytes), dtype=torch.uint8, pin_memory=True)
        else:
            self.layer_cache = LayerCache(
                size=size,
                dtype=torch.uint8,
                shape=(state_nbytes,),
                layer_num=1,
                device="cpu",
                size_first=True,
                shm_name=shm_name + "_states",
                shm_create=shm_create,
            )
            self.buffer = self.layer_cache.buffer[:, 0]

    def get_state_cache(self, buffer_idx):
        layout = self.layout
        data = self.buffer[buffer_idx]
        swa = data[: layout.swa_nbytes].view(layout.layer_num, layout.swa_gpu_pages_per_page, -1)
        c4_start = layout.c4_state_offset - layout.swa_offset
        indexer_start = layout.c4_indexer_state_offset - layout.swa_offset
        c4 = (
            data[c4_start:indexer_start]
            .view(torch.float32)
            .view(layout.n_c4, layout.c4_state_rows, 4 * layout.head_dim)
        )
        indexer = (
            data[indexer_start:]
            .view(torch.float32)
            .view(layout.n_c4, layout.c4_state_rows, 4 * layout.indexer_head_dim)
        )
        return swa, c4, indexer
