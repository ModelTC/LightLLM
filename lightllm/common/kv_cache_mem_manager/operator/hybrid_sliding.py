import torch
import triton

from lightllm.utils.dist_utils import get_current_rank_in_dp, get_dp_world_size

from .normal import NormalMemOperator


class HybridSlidingMemOperator(NormalMemOperator):
    """Full-KV operations and CPU transfers of hybrid sliding checkpoints."""

    def copy_kv_to_mem_manager(self, layer_index: int, mem_index: torch.Tensor, kv: torch.Tensor):
        layer_index = self.mem_manager.sliding_config.full_layer_to_cache_index[layer_index]
        return super().copy_kv_to_mem_manager(layer_index, mem_index, kv)

    def load_cpu_cache_to_gpu(self, mem_indexes, page_indexes, cpu_cache_client, req):
        from lightllm.common.basemodel.triton_kernel.sliding_window_cpu_cache_copy import (
            copy_cpu_cache_to_kv_buffer,
        )
        from lightllm.server.router.model_infer.infer_batch import g_infer_context

        if not len(page_indexes):
            return

        mem_manager = self.mem_manager
        page_size = mem_manager.big_page_token_num
        big_page_num = len(mem_indexes) // page_size
        max_kv_len = (req.cur_kv_len // page_size) * page_size
        big_page_ids = []
        for _ in range(big_page_num):
            page_id = mem_manager.big_page_buffers.alloc_one_state_cache()
            assert page_id is not None
            req.hybrid_len_to_big_page_id[max_kv_len] = page_id
            big_page_ids.append(page_id)
            max_kv_len -= page_size
        big_page_ids.reverse()

        if len(mem_indexes) % page_size:
            padded_token_num = triton.cdiv(len(mem_indexes), page_size) * page_size - len(mem_indexes)
            mem_indexes = torch.nn.functional.pad(mem_indexes, (0, padded_token_num), value=-1)
            # The CPU tail carries a checkpoint before a big-page boundary.
            # Restore through a reserved slot; it must not become a radix big page.
            big_page_ids.append(mem_manager.CPU_CACHE_BIG_PAGE_LOAD_TEMP_BUFFER_ID)

        big_page_ids_gpu = torch.tensor(big_page_ids, dtype=torch.int64, device="cpu").cuda(non_blocking=True)
        copy_cpu_cache_to_kv_buffer(
            mem_indexes=mem_indexes,
            page_indexes=page_indexes,
            big_page_buffer_ids=big_page_ids_gpu,
            gpu_full_att_kv_state=mem_manager.kv_buffer,
            cpu_kv_sliding_state=mem_manager.big_page_buffers.state_cache,
            cpu_cache_tensor=cpu_cache_client.cpu_kv_cache_tensor,
            tp_rank=get_current_rank_in_dp(),
            tp_world_size=get_dp_world_size(),
            big_page_token_num=page_size,
        )
        # Loads and this restore use the inference stream. The next load may
        # reuse its reserved slot only after this copy has been queued.
        g_infer_context.req_manager.restore_big_page_state(big_page_buffer_idx=big_page_ids[-1], req=req)

    def offload_gpu_kv_to_cpu_cache(self, mem_indexes, page_indexes, page_readies, cpu_cache_client, req):
        from lightllm.common.basemodel.triton_kernel.sliding_window_cpu_cache_copy import (
            copy_kv_buffer_to_cpu_cache,
            copy_sliding_window_state,
        )
        from lightllm.server.router.model_infer.infer_batch import g_infer_context

        if not len(page_indexes):
            return

        mem_manager = self.mem_manager
        page_size = mem_manager.big_page_token_num
        radix_cache = g_infer_context.radix_cache
        big_page_ids = radix_cache.get_big_page_ids_by_node(req.shared_kv_node)
        max_kv_len = (len(mem_indexes) // page_size) * page_size
        start_kv_len = (len(big_page_ids) + 1) * page_size
        for seq_len in range(start_kv_len, max_kv_len + 1, page_size):
            big_page_ids.append(req.hybrid_len_to_big_page_id[seq_len])

        if len(mem_indexes) % page_size:
            padded_token_num = triton.cdiv(len(mem_indexes), page_size) * page_size - len(mem_indexes)
            mem_indexes = torch.nn.functional.pad(mem_indexes, (0, padded_token_num), value=-1)
            assert req.tail_small_page_buffer_id is not None
            temp_id = mem_manager.CPU_CACHE_BIG_PAGE_OFFLOAD_TEMP_BUFFER_ID
            src_state = radix_cache.small_page_buffers.get_state_cache(req.tail_small_page_buffer_id)
            copy_sliding_window_state(src_state, mem_manager.big_page_buffers.get_state_cache(temp_id))
            big_page_ids.append(temp_id)

        big_page_ids_gpu = torch.tensor(big_page_ids, dtype=torch.int64, device="cpu").cuda(non_blocking=True)
        # Both staging and the transfer run on the CPU-cache offload stream.
        # Serial stream order protects this slot across requests; load uses a
        # different reserved slot and cannot overwrite an in-flight offload.
        copy_kv_buffer_to_cpu_cache(
            mem_indexes=mem_indexes,
            page_indexes=page_indexes,
            page_readies=page_readies,
            big_page_buffer_ids=big_page_ids_gpu,
            gpu_full_att_kv_state=mem_manager.kv_buffer,
            cpu_kv_sliding_state=mem_manager.big_page_buffers.state_cache,
            cpu_cache_tensor=cpu_cache_client.cpu_kv_cache_tensor,
            tp_rank=get_current_rank_in_dp(),
            tp_world_size=get_dp_world_size(),
            big_page_token_num=page_size,
        )

    def copy_mem_to_mem(self, src_mem_index: torch.Tensor, dst_mem_index: torch.Tensor):
        from lightllm.common.basemodel.triton_kernel.kv_move import copy_kv_buffer_to_kv_buffer

        copy_kv_buffer_to_kv_buffer(
            src_mem_index.cuda(non_blocking=True),
            dst_mem_index.cuda(non_blocking=True),
            self.mem_manager.kv_buffer,
        )
