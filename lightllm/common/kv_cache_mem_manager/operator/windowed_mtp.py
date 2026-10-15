from .hybrid import HybridAttMemOperator
from .normal import NormalMemOperator
from lightllm.utils.dist_utils import get_current_rank_in_dp, get_dp_world_size
from lightllm.common.basemodel.triton_kernel.kv_cache_offload import load_cpu_kv_to_gpu, offload_gpu_kv_to_cpu


class WindowedMTPMemOperator(HybridAttMemOperator, NormalMemOperator):
    """Ordinary target KV plus request-local draft windows in hybrid CPU pages."""

    def _load_target_pages(self, mem_indexes, buffer_ids, page_indexes, cpu_cache_client):
        load_cpu_kv_to_gpu(
            gpu_mem_indexes=mem_indexes,
            gpu_kv_cache=self.mem_manager.kv_buffer,
            gpu_kv_cache_scale=None,
            cpu_kv_cache=self.cpu_cache_config.get_full_att_kv_view(cpu_cache_client.cpu_kv_cache_tensor),
            cpu_kv_cache_scale=None,
            page_indexes=page_indexes,
            tp_index=get_current_rank_in_dp(),
            tp_world_size=get_dp_world_size(),
            grid_num=16,
        )

    def _offload_target_pages(self, mem_indexes, buffer_ids, page_indexes, page_readies, cpu_cache_client):
        offload_gpu_kv_to_cpu(
            token_indexes=mem_indexes,
            gpu_kv_cache=self.mem_manager.kv_buffer,
            gpu_kv_cache_scale=None,
            cpu_kv_cache=self.cpu_cache_config.get_full_att_kv_view(cpu_cache_client.cpu_kv_cache_tensor),
            cpu_kv_cache_scale=None,
            page_indexes=page_indexes,
            page_readies=page_readies,
            tp_index=get_current_rank_in_dp(),
            tp_world_size=get_dp_world_size(),
            grid_num=16,
        )
