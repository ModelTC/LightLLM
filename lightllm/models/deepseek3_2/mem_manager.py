from typing import List, Optional
from typing_extensions import override
import torch

from lightllm.common.kv_cache_mem_manager import MemoryManager
from lightllm.common.kv_cache_mem_manager.deepseek2_mem_manager import Deepseek2MemoryManager
from lightllm.server.pd_io_struct import KVMoveTask
from lightllm.distributed.pynccl import PyNcclCommunicator
from lightllm.common.basemodel.cache_ops import MultiBufferMemoryManager


class Deepseek3_2MemoryManager(Deepseek2MemoryManager, MultiBufferMemoryManager):
    def __init__(
        self, size, dtype, head_num, head_dim, layer_num, always_copy=False, mem_fraction=0.9, is_sub_mem_manager=False
    ):
        super().__init__(size, dtype, head_num, head_dim, layer_num, always_copy, mem_fraction, is_sub_mem_manager)
        self.indexer_ks_mem_manager = Deepseek2MemoryManager(
            self.size, torch.uint8, 1, 132, layer_num, is_sub_mem_manager=True
        )
        return

    # ===== MultiBufferMemoryManager Implementation =====

    def get_aux_buffer_names(self) -> List[str]:
        """
        Return names of auxiliary buffers managed by this memory manager.

        DeepSeek V3.2 has one auxiliary buffer: 'indexer_ks' for NSA top-k selection.
        """
        return ["indexer_ks"]

    def get_aux_buffer(self, name: str) -> Optional[torch.Tensor]:
        """
        Get auxiliary buffer by name.

        Args:
            name: Name of the buffer (only 'indexer_ks' is supported)

        Returns:
            Buffer tensor or None if not found
        """
        if name == "indexer_ks":
            return self.indexer_ks_mem_manager.kv_buffer
        return None

    def copy_aux_buffer_tokens(
        self,
        buffer_name: str,
        layer_idx: int,
        src_positions: torch.Tensor,
        dest_positions: torch.Tensor,
    ) -> None:
        """
        Copy tokens in auxiliary buffer from source to destination positions.

        This method provides a generic interface for buffer copying, which
        can be used by the sync_prefill_buffers hook.

        Args:
            buffer_name: Name of auxiliary buffer (only 'indexer_ks' is supported)
            layer_idx: Layer index
            src_positions: Source token positions [N]
            dest_positions: Destination token positions [N]
        """
        if buffer_name != "indexer_ks":
            raise ValueError(f"Unknown buffer: {buffer_name}. Supported buffers: ['indexer_ks']")

        from lightllm.models.deepseek3_2.triton_kernel.copy_indexer_ks import copy_indexer_ks

        buffer = self.indexer_ks_mem_manager.kv_buffer[layer_idx]
        copy_indexer_ks(buffer=buffer, src_loc=src_positions, dest_loc=dest_positions)

    @override
    def get_cell_size(self):
        return super().get_cell_size() + 132


class Deepseek3_2FP8KVMemoryManager(Deepseek3_2MemoryManager):
    def __init__(
        self, size, dtype, head_num, head_dim, layer_num, always_copy=False, mem_fraction=0.9, is_sub_mem_manager=False
    ):
        super().__init__(
            size, torch.uint8, head_num, head_dim + 2, layer_num, always_copy, mem_fraction, is_sub_mem_manager
        )
