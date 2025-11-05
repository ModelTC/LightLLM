import os
import torch
import numpy as np
import torch.distributed as dist
from lightllm.models.deepseek2.flashattention_infer_struct import Deepseek2FlashAttentionStateInfo
from lightllm.models.deepseek3_2.mem_manager import Deepseek3_2IndexerPagedMemoryManager, Deepseek3_2MemoryManager


class Deepseek3_2FlashAttentionInferStateInfo(Deepseek2FlashAttentionStateInfo):
    
    def __init__(self):
        super().__init__()
        assert isinstance(self.req_manager.mem_manager, Deepseek3_2MemoryManager)
        self.indexer_paged_mem_manager : Deepseek3_2IndexerPagedMemoryManager = self.req_manager.mem_manager.indexer_paged_mem_manager
