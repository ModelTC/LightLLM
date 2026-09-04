import torch

from .normal import NormalMemOperator


class HybridSlidingMemOperator(NormalMemOperator):
    """GPU KV operations for the token-granular full-attention cache."""

    def copy_mem_to_mem(self, src_mem_index: torch.Tensor, dst_mem_index: torch.Tensor):
        from lightllm.common.basemodel.triton_kernel.kv_move import copy_kv_buffer_to_kv_buffer

        copy_kv_buffer_to_kv_buffer(
            src_mem_index.cuda(non_blocking=True),
            dst_mem_index.cuda(non_blocking=True),
            self.mem_manager.kv_buffer,
        )
