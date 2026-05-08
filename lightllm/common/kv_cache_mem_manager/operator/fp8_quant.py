import torch
from typing import TYPE_CHECKING
from .base import BaseMemManagerOperator
from lightllm.utils.log_utils import init_logger

if TYPE_CHECKING:
    from lightllm.server.multi_level_kv_cache.cpu_cache_client import CpuKvCacheClient
logger = init_logger(__name__)


class FP8StaticPerHeadQuantMemOperator(BaseMemManagerOperator):
    def copy_kv_to_mem_manager(self, layer_index: int, mem_index: torch.Tensor, kv: torch.Tensor):
        from lightllm.common.kv_cache_mem_manager.fp8_static_per_head_quant_mem_manager import (
            FP8StaticPerHeadQuantMemManager,
        )

        mem_manager: FP8StaticPerHeadQuantMemManager = self.mem_manager
        from lightllm.common.basemodel.triton_kernel.destindex_copy_kv_fp8 import (
            destindex_copy_kv_fp8,
        )

        scales = mem_manager.scales
        destindex_copy_kv_fp8(
            kv,
            mem_index,
            scales[layer_index],
            mem_manager.kv_buffer[layer_index].view(torch.float8_e4m3fn),
        )
        return


class FP8StaticPerTensorQuantMemOperator(BaseMemManagerOperator):
    def copy_kv_to_mem_manager(self, layer_index: int, mem_index: torch.Tensor, kv: torch.Tensor):
        from lightllm.common.kv_cache_mem_manager.mem_manager import MemoryManager

        mem_manager: MemoryManager = self.mem_manager
        from lightllm.common.basemodel.triton_kernel.destindex_copy_kv_fp8 import (
            destindex_copy_kv_fp8,
        )

        destindex_copy_kv_fp8(
            kv,
            mem_index,
            mem_manager.scales[layer_index],
            mem_manager.kv_buffer[layer_index].view(torch.float8_e4m3fn),
            is_per_tensor_quant=True,
        )
        return
