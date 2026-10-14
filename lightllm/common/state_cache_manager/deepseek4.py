import math

import torch

from .base import StateCacheManager


class DeepseekV4StateCacheManager(StateCacheManager):
    """Pinned continuation checkpoints; compressed history stays in token pages."""

    def __init__(self, size, *, swa_shape, c4_shape, indexer_shape, keep_num=0):
        super().__init__(size, keep_num)
        self.swa_shape = swa_shape
        self.c4_shape = c4_shape
        self.indexer_shape = indexer_shape
        self.swa_nbytes = math.prod(swa_shape)
        self.c4_nbytes = math.prod(c4_shape) * torch.float32.itemsize
        indexer_nbytes = math.prod(indexer_shape) * torch.float32.itemsize
        self.buffer = torch.empty(
            (size, self.swa_nbytes + self.c4_nbytes + indexer_nbytes), dtype=torch.uint8, pin_memory=True
        )

    def get_state_cache(self, buffer_idx):
        data = self.buffer[buffer_idx]
        swa = data[: self.swa_nbytes].view(self.swa_shape)
        indexer_start = self.swa_nbytes + self.c4_nbytes
        c4 = data[self.swa_nbytes : indexer_start].view(torch.float32).view(self.c4_shape)
        indexer = data[indexer_start:].view(torch.float32).view(self.indexer_shape)
        return swa, c4, indexer
