import torch

from .base import StateCacheManager


class DeepseekV4StateCacheManager(StateCacheManager):
    """Pinned continuation checkpoints; compressed history stays in token pages."""

    def __init__(self, size, layout, keep_num=0):
        super().__init__(size, keep_num)
        self.layout = layout
        self.buffer = torch.empty((size, layout.page_nbytes - layout.swa_offset), dtype=torch.uint8, pin_memory=True)

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
