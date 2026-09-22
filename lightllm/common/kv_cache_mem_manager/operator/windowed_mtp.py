from .hybrid import HybridAttMemOperator
from .normal import NormalMemOperator
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.dist_utils import get_current_rank_in_dp
from lightllm.common.basemodel.triton_kernel.hybrid_cpu_cache_copy import copy_full_att_pages


class WindowedMTPMemOperator(HybridAttMemOperator, NormalMemOperator):
    """Ordinary target KV plus request-local draft windows in hybrid CPU pages."""

    def _load_target_pages(self, mem_indexes, buffer_ids, page_indexes, cpu_cache_client):
        copy_full_att_pages(
            self.mem_manager.kv_buffer,
            cpu_cache_client.cpu_kv_cache_tensor,
            mem_indexes,
            page_indexes,
            self.cpu_cache_config,
            get_current_rank_in_dp(),
            get_env_start_args().cpu_cache_token_page_size,
        )

    def _offload_target_pages(self, mem_indexes, buffer_ids, page_indexes, page_readies, cpu_cache_client):
        copy_full_att_pages(
            self.mem_manager.kv_buffer,
            cpu_cache_client.cpu_kv_cache_tensor,
            mem_indexes,
            page_indexes,
            self.cpu_cache_config,
            get_current_rank_in_dp(),
            get_env_start_args().cpu_cache_token_page_size,
            page_readies=page_readies,
        )
