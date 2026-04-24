import torch
from typing import TYPE_CHECKING, List
from .base import BaseMemManagerOperator
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.dist_utils import get_current_rank_in_dp, get_dp_world_size
from lightllm.utils.log_utils import init_logger

if TYPE_CHECKING:
    from lightllm.server.multi_level_kv_cache.cpu_cache_client import CpuKvCacheClient
    from lightllm.server.router.model_infer.infer_batch import InferReq

logger = init_logger(__name__)


class QuantScaleMemOperator(BaseMemManagerOperator):
    """
    对于kv cache中包含独立的对应每个token的scale变量的memManager使用。
    """

    def load_cpu_kv_to_gpu(
        self,
        mem_indexes: torch.Tensor,
        page_indexes: torch.Tensor,
        cpu_cache_client: "CpuKvCacheClient",
        req: "InferReq",
    ):
        assert mem_indexes.is_cuda and page_indexes.is_cuda
        args = get_env_start_args()
        assert len(mem_indexes) % args.cpu_cache_token_page_size == 0
        from lightllm.common.kv_cache_mem_manager.mem_manager import MemoryManager

        mem_manager: MemoryManager = self.mem_manager

        cpu_cache_meta = cpu_cache_client.kv_cache_tensor_meta
        cpu_kv_cache = cpu_cache_client.cpu_kv_cache_tensor[:, :, :, :, 0 : cpu_cache_meta.head_dim]
        cpu_kv_cache_scale = cpu_cache_client.cpu_kv_cache_tensor[:, :, :, :, cpu_cache_meta.head_dim :].view(
            mem_manager.scale_buffer.dtype
        )

        from lightllm.common.basemodel.triton_kernel.kv_cache_offload import load_cpu_kv_to_gpu

        load_cpu_kv_to_gpu(
            gpu_mem_indexes=mem_indexes,
            gpu_kv_cache=mem_manager.kv_buffer,
            gpu_kv_cache_scale=mem_manager.scale_buffer,
            cpu_kv_cache=cpu_kv_cache,
            cpu_kv_cache_scale=cpu_kv_cache_scale,
            tp_index=get_current_rank_in_dp(),
            tp_world_size=get_dp_world_size(),
            grid_num=16,
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

        cpu_cache_meta = cpu_cache_client.kv_cache_tensor_meta
        cpu_kv_cache = cpu_cache_client.cpu_kv_cache_tensor[:, :, :, :, 0 : cpu_cache_meta.head_dim]
        cpu_kv_cache_scale = cpu_cache_client.cpu_kv_cache_tensor[:, :, :, :, cpu_cache_meta.head_dim :].view(
            mem_manager.scale_buffer.dtype
        )

        from lightllm.common.basemodel.triton_kernel.kv_cache_offload import offload_gpu_kv_to_cpu

        offload_gpu_kv_to_cpu(
            token_indexes=mem_indexes,
            gpu_kv_cache=mem_manager.kv_buffer,
            gpu_kv_cache_scale=mem_manager.scale_buffer,
            cpu_kv_cache=cpu_kv_cache,
            cpu_kv_cache_scale=cpu_kv_cache_scale,
            page_indexes=page_indexes,
            page_readies=page_readies,
            tp_index=get_current_rank_in_dp(),
            tp_world_size=get_dp_world_size(),
            grid_num=16,
        )
        return


class PPLInt4KVMemOperator(QuantScaleMemOperator):
    def copy_kv_to_mem_manager(self, layer_index: int, mem_index: torch.Tensor, kv: torch.Tensor):
        from lightllm.common.kv_cache_mem_manager.ppl_int4kv_mem_manager import PPLINT4KVMemoryManager

        mem_manager: PPLINT4KVMemoryManager = self.mem_manager
        from ...basemodel.triton_kernel.kv_copy.ppl_int4kv_copy_kv import (
            destindex_copy_int4kv,
        )

        destindex_copy_int4kv(
            kv,
            mem_index,
            mem_manager.kv_buffer[layer_index],
            mem_manager.scale_buffer[layer_index],
            quant_group_size=mem_manager.group_quant_size,
        )
        return


class PPLInt8KVMemOperator(QuantScaleMemOperator):
    def copy_kv_to_mem_manager(self, layer_index: int, mem_index: torch.Tensor, kv: torch.Tensor):
        from lightllm.common.kv_cache_mem_manager.ppl_int8kv_mem_manager import PPLINT8KVMemoryManager

        mem_manager: PPLINT8KVMemoryManager = self.mem_manager
        from ...basemodel.triton_kernel.kv_copy.ppl_int8kv_copy_kv import (
            destindex_copy_quantize_kv,
        )

        destindex_copy_quantize_kv(
            kv,
            mem_index,
            mem_manager.kv_buffer[layer_index],
            mem_manager.scale_buffer[layer_index],
            quant_group_dim=mem_manager.group_quant_size,
        )
        return
