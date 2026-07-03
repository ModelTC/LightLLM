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
