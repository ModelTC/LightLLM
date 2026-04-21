import torch
from typing import TYPE_CHECKING
from .base import BaseMemManagerOperator
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.dist_utils import get_current_rank_in_dp, get_dp_world_size
from lightllm.utils.log_utils import init_logger
from lightllm.common.linear_att_cache_manager.config_objs import LinearAttCacheConfig

if TYPE_CHECKING:
    from lightllm.server.multi_level_kv_cache.cpu_cache_client import CpuKvCacheClient
logger = init_logger(__name__)


class LinearAttMemOperator(BaseMemManagerOperator):
    """
    只用于非量化的linear att 混合 full att的模型，列入 qwen3.5
    """

    def __init__(self, mem_manager):
        super().__init__(mem_manager)
        self.linear_config = LinearAttCacheConfig.load_from_args()

    def load_cpu_kv_to_gpu(
        self, mem_indexes: torch.Tensor, page_indexes: torch.Tensor, cpu_cache_client: "CpuKvCacheClient"
    ):
        assert mem_indexes.is_cuda and page_indexes.is_cuda
        args = get_env_start_args()
        assert len(mem_indexes) % args.cpu_cache_token_page_size == 0
        from lightllm.common.kv_cache_mem_manager.mem_manager import MemoryManager

        mem_manager: MemoryManager = self.mem_manager

        from lightllm.common.basemodel.triton_kernel.linear_att_cpu_cache_copy import (
            copy_cpu_cache_to_kv_buffer,
        )

        copy_cpu_cache_to_kv_buffer(
            mem_indexes=mem_indexes,
            page_indexes=page_indexes,
            gpu_full_att_kv_state=mem_manager.kv_buffer,
            cpu_kv_conv_state=mem_manager.conv_state_buffer,
            cpu_kv_ssm_state=mem_manager.ssm_state_buffer,
            cpu_cache_tensor=cpu_cache_client.cpu_kv_cache_tensor,
            tp_rank=get_current_rank_in_dp(),
            tp_world_size=get_dp_world_size(),
            big_page_token_num=args.cpu_cache_token_page_size,
            linear_config=self.linear_config,
        )
        return

    def offload_gpu_kv_to_cpu(
        self,
        mem_indexes: torch.Tensor,
        page_indexes: torch.Tensor,
        page_readies: torch.Tensor,
        cpu_cache_client: "CpuKvCacheClient",
    ):
        assert mem_indexes.is_cuda and page_indexes.is_cuda and page_readies.is_cuda
        args = get_env_start_args()
        assert len(mem_indexes) % args.cpu_cache_token_page_size == 0
        assert len(mem_indexes) // args.cpu_cache_token_page_size == len(page_indexes)
        from lightllm.common.kv_cache_mem_manager.mem_manager import MemoryManager

        mem_manager: MemoryManager = self.mem_manager

        from lightllm.common.basemodel.triton_kernel.linear_att_cpu_cache_copy import (
            copy_kv_buffer_to_cpu_cache,
        )

        copy_kv_buffer_to_cpu_cache(
            mem_indexes=mem_indexes,
            page_indexes=page_indexes,
            page_readies=page_readies,
            gpu_full_att_kv_state=mem_manager.kv_buffer,
            cpu_kv_conv_state=mem_manager.conv_state_buffer,
            cpu_kv_ssm_state=mem_manager.ssm_state_buffer,
            cpu_cache_tensor=cpu_cache_client.cpu_kv_cache_tensor,
            tp_rank=get_current_rank_in_dp(),
            tp_world_size=get_dp_world_size(),
            big_page_token_num=args.cpu_cache_token_page_size,
            linear_config=self.linear_config,
        )
        return
