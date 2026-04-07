import torch
from typing import Any

from .deepseek2_mem_manager import Deepseek2MemoryManager


class FP8PerTokenGroupQuantDeepseek3_2MemoryManager(Deepseek2MemoryManager):
    kv_nope_dim = 512
    kv_rope_dim = 64
    # 576 = 512 + 64
    kv_head_dim = kv_nope_dim + kv_rope_dim

    quant_group_size = 128
    # 4 = 512 / 128
    quant_group_num = kv_nope_dim // quant_group_size
    # 4 * 4 = quant_group_num * fp32
    # 64 * 2 = kv_rope_dim * bfloat16
    # 656 bytes = 512 + (4 * 4) + (64 * 2)
    flashmla_bytes_per_token = kv_nope_dim + quant_group_num * 4 + kv_rope_dim * 2

    indexer_head_dim = 128
    # 128 + 4 = indexer_head_dim + fp32
    # 132 bytes = 128 + 4
    indexer_bytes_per_token = indexer_head_dim + 4

    # 16-byte 对齐，满足FlashMLA的对齐要求
    alignment = 16
    total_bytes_per_token = (
        (flashmla_bytes_per_token + indexer_bytes_per_token + alignment - 1) // alignment * alignment
    )

    def __init__(self, size, dtype, head_num, head_dim, layer_num, always_copy=False, mem_fraction=0.9):
        assert head_num == 1, "DeepSeek-V3.2 DSA FP8 path expects MQA-style head_num == 1"
        self.prefill_dtype = dtype
        super().__init__(size, torch.uint8, head_num, self.total_bytes_per_token, layer_num, always_copy, mem_fraction)

    def copy_kv_to_mem_manager(self, layer_index: int, mem_index: torch.Tensor, kv: torch.Tensor):
        from lightllm.models.deepseek3_2.triton_kernel.destindex_copy_kv_flashmla_fp8 import (
            destindex_copy_kv_flashmla_fp8,
        )

        rope_dim = 64
        kv_lora_rank = kv.shape[2] - rope_dim
        assert kv_lora_rank == 512, f"Expected kv_lora_rank=512, got {kv_lora_rank}"

        o_nope = self.kv_buffer[layer_index][:, :, :512].view(torch.float8_e4m3fn)
        o_scale = self.kv_buffer[layer_index][:, :, 512:528].view(torch.float32)
        o_rope = self.kv_buffer[layer_index][:, :, 528 : self.flashmla_bytes_per_token].view(torch.bfloat16)
        destindex_copy_kv_flashmla_fp8(
            kv[:, :, :kv_lora_rank],
            kv[:, :, kv_lora_rank:],
            mem_index,
            o_nope,
            o_scale,
            o_rope,
        )

    def get_att_input_params(self, layer_index: int) -> Any:
        return self.kv_buffer[layer_index][:, :, : self.flashmla_bytes_per_token]

    def get_indexer_k_buffer(self, layer_index: int) -> torch.Tensor:
        begin = self.flashmla_bytes_per_token
        end = begin + self.indexer_bytes_per_token
        return self.kv_buffer[layer_index][:, :, begin:end]

    def get_prefill_kv_cache_and_remap_indices(
        self,
        packed_kv: torch.Tensor,
        topk_indices: torch.Tensor,
        prefill_mem_index: torch.Tensor,
        prefill_cache_kv: torch.Tensor,
    ):
        from lightllm.models.deepseek3_2.triton_kernel.prefill_compact_kv_flashmla_fp8 import (
            get_prefill_kv_cache_and_remap_indices_triton,
        )

        return get_prefill_kv_cache_and_remap_indices_triton(
            packed_kv=packed_kv,
            topk_mem_indices=topk_indices,
            prefill_mem_index=prefill_mem_index,
            prefill_cache_kv=prefill_cache_kv,
            prefill_dtype=self.prefill_dtype,
        )
