import torch
from .base import BaseMemManagerOperator


class DeepseekV4MemOperator(BaseMemManagerOperator):
    def copy_mem_to_mem(self, src_mem_index: torch.Tensor, dst_mem_index: torch.Tensor):
        """Copy exactly the requested token slots; request state has a separate owner."""
        from lightllm.common.basemodel.triton_kernel.kv_move import copy_kv_buffer_to_kv_buffer

        # The coordinator restores SWA and compressor state separately.
        src = src_mem_index.to(device="cuda", dtype=torch.int64, non_blocking=True)
        dst = dst_mem_index.to(device="cuda", dtype=torch.int64, non_blocking=True)
        for buffer in self.mem_manager.kv_buffer:
            copy_kv_buffer_to_kv_buffer(src, dst, buffer.view(1, buffer.shape[0], -1))

    def copy_kv_to_mem_manager(self, layer_index: int, mem_index: torch.Tensor, kv: torch.Tensor):
        # _get_qkv() writes through the state manager.
        # One kernel runs RMSNorm, RoPE, FP8 packing, and SWA writes.
        # This avoids extra tensors and kernel launches.
        # Prefill and decode skip _post_cache_kv(). They do not call this method.
        # The compressor and indexer write history to mem_manager.kv_buffer.
        raise NotImplementedError("DeepSeek-V4 writes packed KV using request-owned SWA slots")

    def copy_kv_from_other_dp_ranks(
        self, kv_views, move_token_indexes, token_dp_indexes, mem_indexes, dp_size_in_node, rank_in_dp
    ):
        from lightllm.common.kv_trans_kernel.kv_trans_v2 import kv_trans_for_dp

        for layer, buffer in enumerate(self.mem_manager.kv_buffer):
            pointers = torch.tensor(
                [view.kv_buffer[layer].data_ptr() for view in kv_views], dtype=torch.uint64, device="cuda"
            )
            kv_trans_for_dp(
                pointers,
                move_token_indexes,
                token_dp_indexes,
                buffer.unsqueeze(0),
                mem_indexes,
                dp_size_in_node,
                rank_in_dp,
            )
