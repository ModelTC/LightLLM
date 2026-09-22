import torch
from .hybrid import HybridAttMemOperator
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.dist_utils import get_current_rank_in_dp, get_dp_world_size
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class LinearAttMemOperator(HybridAttMemOperator):
    """
    只用于非量化的linear att 混合 full att的模型，列入 qwen3.5
    """

    def __init__(self, mem_manager):
        super().__init__(mem_manager)
        self.linear_config = mem_manager.linear_config

    def _load_target_pages(self, mem_indexes, big_page_buffer_ids_gpu, page_indexes, cpu_cache_client):
        args = get_env_start_args()
        mem_manager = self.mem_manager
        from lightllm.common.basemodel.triton_kernel.linear_att_cpu_cache_copy import (
            copy_cpu_cache_to_kv_buffer,
        )

        copy_cpu_cache_to_kv_buffer(
            mem_indexes=mem_indexes,
            big_page_buffer_ids=big_page_buffer_ids_gpu,
            page_indexes=page_indexes,
            gpu_full_att_kv_state=mem_manager.kv_buffer,
            cpu_kv_conv_state=mem_manager.big_page_buffers.conv_state_cache.buffer,
            cpu_kv_ssm_state=mem_manager.big_page_buffers.ssm_state_cache.buffer,
            cpu_cache_tensor=cpu_cache_client.cpu_kv_cache_tensor,
            tp_rank=get_current_rank_in_dp(),
            tp_world_size=get_dp_world_size(),
            big_page_token_num=args.cpu_cache_token_page_size,
            linear_config=self.linear_config,
        )

    def _offload_target_pages(self, mem_indexes, big_page_buffer_ids_gpu, page_indexes, page_readies, cpu_cache_client):
        args = get_env_start_args()
        mem_manager = self.mem_manager
        from lightllm.common.basemodel.triton_kernel.linear_att_cpu_cache_copy import (
            copy_kv_buffer_to_cpu_cache,
        )

        copy_kv_buffer_to_cpu_cache(
            mem_indexes=mem_indexes,
            page_indexes=page_indexes,
            page_readies=page_readies,
            big_page_buffer_ids=big_page_buffer_ids_gpu,
            gpu_kv_full_att_state=mem_manager.kv_buffer,
            cpu_kv_conv_state=mem_manager.big_page_buffers.conv_state_cache.buffer,
            cpu_kv_ssm_state=mem_manager.big_page_buffers.ssm_state_cache.buffer,
            cpu_cache_tensor=cpu_cache_client.cpu_kv_cache_tensor,
            tp_rank=get_current_rank_in_dp(),
            tp_world_size=get_dp_world_size(),
            big_page_token_num=args.cpu_cache_token_page_size,
            linear_config=self.linear_config,
        )

    def _copy_target_state(self, source, source_id, destination_id):
        from lightllm.common.basemodel.triton_kernel.linear_att_cpu_cache_copy import (
            copy_linear_att_state_to_linear_att_state,
        )

        src_conv_state, src_ssm_state = source.get_state_cache(source_id)
        dst_conv_state, dst_ssm_state = self.mem_manager.big_page_buffers.get_state_cache(destination_id)
        copy_linear_att_state_to_linear_att_state(
            src_conv_state=src_conv_state,
            src_ssm_state=src_ssm_state,
            dst_conv_state=dst_conv_state,
            dst_ssm_state=dst_ssm_state,
        )

    def copy_kv_to_mem_manager(self, layer_index: int, mem_index: torch.Tensor, kv: torch.Tensor):
        # Qwen3Next 需要调整 layer_index
        layer_index = self.linear_config.get_full_att_kv_layer_index(layer_index)
        from lightllm.common.kv_cache_mem_manager.mem_manager import MemoryManager

        mem_manager: MemoryManager = self.mem_manager
        from lightllm.common.basemodel.triton_kernel.destindex_copy_kv import (
            destindex_copy_kv,
        )

        destindex_copy_kv(kv, mem_index, mem_manager.kv_buffer[layer_index])
        return
