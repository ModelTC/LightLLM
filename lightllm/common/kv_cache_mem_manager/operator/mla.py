import torch
from .normal import NormalMemOperator
from .base import BaseMemManagerOperator
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class MlaMemOperator(NormalMemOperator):
    def copy_kv_to_mem_manager(self, layer_index, mem_index, kv):
        from ...basemodel.triton_kernel.kv_copy.mla_copy_kv import destindex_copy_kv

        output = self.mem_manager.get_att_input_params(layer_index)
        rope_dim = 64
        kv_lora_rank = kv.shape[2] - rope_dim
        assert kv.shape[2] == output.shape[-1]
        destindex_copy_kv(
            kv[:, :, :kv_lora_rank],
            kv[:, :, kv_lora_rank:],
            mem_index,
            output[:, :, :kv_lora_rank],
            output[:, :, kv_lora_rank:],
        )


class FP8IndexedMlaMemOperator(BaseMemManagerOperator):
    def copy_kv_to_mem_manager(self, layer_index: int, mem_index: torch.Tensor, kv: torch.Tensor):
        layer_index = self.mem_manager.get_kv_layer_index(layer_index)
        from lightllm.common.kv_cache_mem_manager.mla_mem_manager import (
            FP8IndexedMlaMemoryManager,
        )

        mem_manager: FP8IndexedMlaMemoryManager = self.mem_manager
        from lightllm.common.basemodel.triton_kernel.kv_copy.destindex_copy_kv_flashmla_fp8 import (
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
