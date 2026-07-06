import torch
from .normal import NormalMemOperator
from .base import BaseMemManagerOperator
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class Deepseek2MemOperator(NormalMemOperator):
    def copy_kv_to_mem_manager(self, layer_index: int, mem_index: torch.Tensor, kv: torch.Tensor):
        from lightllm.common.kv_cache_mem_manager.deepseek2_mem_manager import Deepseek2MemoryManager

        mem_manager: Deepseek2MemoryManager = self.mem_manager

        from ...basemodel.triton_kernel.kv_copy.mla_copy_kv import destindex_copy_kv

        rope_dim = 64
        kv_lora_rank = kv.shape[2] - rope_dim
        assert kv_lora_rank + rope_dim == mem_manager.kv_buffer.shape[-1]

        destindex_copy_kv(
            kv[:, :, :kv_lora_rank],
            kv[:, :, kv_lora_rank:],
            mem_index,
            mem_manager.kv_buffer[layer_index][:, :, :kv_lora_rank],
            mem_manager.kv_buffer[layer_index][:, :, kv_lora_rank:],
        )
        return


class Deepseek3_2MemOperator(Deepseek2MemOperator):
    def copy_kv_to_mem_manager(self, layer_index: int, mem_index: torch.Tensor, kv: torch.Tensor):
        from lightllm.common.kv_cache_mem_manager.deepseek3_2mem_manager import Deepseek3_2MemoryManager

        mem_manager: Deepseek3_2MemoryManager = self.mem_manager
        from ...basemodel.triton_kernel.kv_copy.mla_copy_kv import destindex_copy_kv

        rope_dim = 64
        kv_lora_rank = kv.shape[2] - rope_dim
        assert kv_lora_rank + rope_dim == mem_manager.kv_buffer.shape[-1] - (144 // 2)

        destindex_copy_kv(
            kv[:, :, :kv_lora_rank],
            kv[:, :, kv_lora_rank:],
            mem_index,
            mem_manager.kv_buffer[layer_index][:, :, :kv_lora_rank],
            mem_manager.kv_buffer[layer_index][:, :, kv_lora_rank : (kv_lora_rank + rope_dim)],
        )
        return


class FP8PerTokenGroupQuantDeepseek3_2MemOperator(BaseMemManagerOperator):
    def _get_cpu_cache_views(self, cpu_cache_client):
        from lightllm.common.kv_cache_mem_manager.fp8_per_token_group_quant_deepseek3_2mem_manager import (
            FP8PerTokenGroupQuantDeepseek3_2MemoryManager,
        )

        mem_manager: FP8PerTokenGroupQuantDeepseek3_2MemoryManager = self.mem_manager
        cpu_cache_meta = cpu_cache_client.kv_cache_tensor_meta
        split = mem_manager.flashmla_bytes_per_token
        end = split + mem_manager.indexer_bytes_per_token
        assert cpu_cache_meta.data_type is torch.uint8
        assert cpu_cache_meta.num_heads == 1
        assert cpu_cache_meta.head_dim >= end

        cpu_cache_tensor = cpu_cache_client.cpu_kv_cache_tensor
        return cpu_cache_tensor[:, :, :, :, :split], cpu_cache_tensor[:, :, :, :, split:end]

    def copy_kv_to_mem_manager(self, layer_index: int, mem_index: torch.Tensor, kv: torch.Tensor):
        from lightllm.common.kv_cache_mem_manager.fp8_per_token_group_quant_deepseek3_2mem_manager import (
            FP8PerTokenGroupQuantDeepseek3_2MemoryManager,
        )

        mem_manager: FP8PerTokenGroupQuantDeepseek3_2MemoryManager = self.mem_manager
        from lightllm.models.deepseek3_2.triton_kernel.destindex_copy_kv_flashmla_fp8 import (
            destindex_copy_kv_flashmla_fp8,
        )

        rope_dim = 64
        kv_lora_rank = kv.shape[2] - rope_dim
        assert kv_lora_rank == 512, f"Expected kv_lora_rank=512, got {kv_lora_rank}"

        flashmla_bytes_per_token = mem_manager.flashmla_bytes_per_token

        o_nope = mem_manager.kv_buffer[layer_index][:, :, :512].view(torch.float8_e4m3fn)
        o_scale = mem_manager.kv_buffer[layer_index][:, :, 512:528].view(torch.float32)
        o_rope = mem_manager.kv_buffer[layer_index][:, :, 528:flashmla_bytes_per_token].view(torch.bfloat16)
        destindex_copy_kv_flashmla_fp8(
            kv[:, :, :kv_lora_rank],
            kv[:, :, kv_lora_rank:],
            mem_index,
            o_nope,
            o_scale,
            o_rope,
        )
        return

    def load_cpu_cache_to_gpu(self, mem_indexes: torch.Tensor, page_indexes: torch.Tensor, cpu_cache_client, req):
        assert mem_indexes.is_cuda and page_indexes.is_cuda
        from lightllm.utils.envs_utils import get_env_start_args
        from lightllm.utils.dist_utils import get_current_rank_in_dp, get_dp_world_size
        from lightllm.common.kv_cache_mem_manager.fp8_per_token_group_quant_deepseek3_2mem_manager import (
            FP8PerTokenGroupQuantDeepseek3_2MemoryManager,
        )
        from lightllm.common.basemodel.triton_kernel.kv_cache_offload import load_cpu_kv_to_gpu

        args = get_env_start_args()
        assert len(mem_indexes) % args.cpu_cache_token_page_size == 0
        mem_manager: FP8PerTokenGroupQuantDeepseek3_2MemoryManager = self.mem_manager
        cpu_kv_cache, cpu_indexer_k_cache = self._get_cpu_cache_views(cpu_cache_client)

        rank_in_dp = get_current_rank_in_dp()
        dp_world_size = get_dp_world_size()
        load_cpu_kv_to_gpu(
            gpu_mem_indexes=mem_indexes,
            gpu_kv_cache=mem_manager.kv_buffer,
            gpu_kv_cache_scale=None,
            cpu_kv_cache=cpu_kv_cache,
            cpu_kv_cache_scale=None,
            page_indexes=page_indexes,
            tp_index=rank_in_dp,
            tp_world_size=dp_world_size,
            grid_num=16,
        )
        load_cpu_kv_to_gpu(
            gpu_mem_indexes=mem_indexes,
            gpu_kv_cache=mem_manager.indexer_k_buffer,
            gpu_kv_cache_scale=None,
            cpu_kv_cache=cpu_indexer_k_cache,
            cpu_kv_cache_scale=None,
            page_indexes=page_indexes,
            tp_index=rank_in_dp,
            tp_world_size=dp_world_size,
            grid_num=16,
        )
        return

    def offload_gpu_kv_to_cpu_cache(
        self,
        mem_indexes: torch.Tensor,
        page_indexes: torch.Tensor,
        page_readies: torch.Tensor,
        cpu_cache_client,
        req,
    ):
        assert mem_indexes.is_cuda and page_indexes.is_cuda and page_readies.is_cuda
        from lightllm.utils.envs_utils import get_env_start_args
        from lightllm.utils.dist_utils import get_current_rank_in_dp, get_dp_world_size
        from lightllm.common.kv_cache_mem_manager.fp8_per_token_group_quant_deepseek3_2mem_manager import (
            FP8PerTokenGroupQuantDeepseek3_2MemoryManager,
        )
        from lightllm.common.basemodel.triton_kernel.kv_cache_offload import offload_gpu_kv_to_cpu

        args = get_env_start_args()
        assert len(mem_indexes) % args.cpu_cache_token_page_size == 0
        assert len(mem_indexes) // args.cpu_cache_token_page_size == len(page_indexes)
        mem_manager: FP8PerTokenGroupQuantDeepseek3_2MemoryManager = self.mem_manager
        cpu_kv_cache, cpu_indexer_k_cache = self._get_cpu_cache_views(cpu_cache_client)

        rank_in_dp = get_current_rank_in_dp()
        dp_world_size = get_dp_world_size()
        offload_gpu_kv_to_cpu(
            token_indexes=mem_indexes,
            gpu_kv_cache=mem_manager.kv_buffer,
            gpu_kv_cache_scale=None,
            cpu_kv_cache=cpu_kv_cache,
            cpu_kv_cache_scale=None,
            page_indexes=page_indexes,
            page_readies=page_readies,
            tp_index=rank_in_dp,
            tp_world_size=dp_world_size,
            grid_num=16,
        )
        offload_gpu_kv_to_cpu(
            token_indexes=mem_indexes,
            gpu_kv_cache=mem_manager.indexer_k_buffer,
            gpu_kv_cache_scale=None,
            cpu_kv_cache=cpu_indexer_k_cache,
            cpu_kv_cache_scale=None,
            page_indexes=page_indexes,
            page_readies=page_readies,
            tp_index=rank_in_dp,
            tp_world_size=dp_world_size,
            grid_num=16,
        )
        return
