import torch
from typing import Any

from .deepseek2_mem_manager import Deepseek2MemoryManager
from .operator import FP8PerTokenGroupQuantDeepseek3_2MemOperator


class FP8PerTokenGroupQuantDeepseek3_2MemoryManager(Deepseek2MemoryManager):

    operator_class = FP8PerTokenGroupQuantDeepseek3_2MemOperator

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

    def __init__(self, size, dtype, head_num, head_dim, layer_num, always_copy=False, mem_fraction=0.9):
        assert head_num == 1, "DeepSeek-V3.2 DSA FP8 path expects MQA-style head_num == 1"
        self.prefill_dtype = dtype
        super().__init__(size, torch.uint8, head_num, self.flashmla_bytes_per_token, layer_num, always_copy, mem_fraction)

    def get_cell_size(self):
        return self.layer_num * (self.flashmla_bytes_per_token + self.indexer_bytes_per_token)

    def _init_buffers(self, size, dtype, head_num, head_dim, layer_num):
        self.kv_buffer = torch.empty(
            (layer_num, size + 1, head_num, self.flashmla_bytes_per_token),
            dtype=dtype,
            device="cuda",
        )
        self.indexer_k_buffer = torch.empty(
            (layer_num, size + 1, head_num, self.indexer_bytes_per_token),
            dtype=dtype,
            device="cuda",
        )

    def get_att_input_params(self, layer_index: int) -> Any:
        return self.kv_buffer[layer_index]

    def get_indexer_k_buffer(self, layer_index: int) -> torch.Tensor:
        return self.indexer_k_buffer[layer_index]

    def _free_buffers(self):
        self.kv_buffer = None
        self.indexer_k_buffer = None

    def get_index_kv_buffer(self, index):
        return {"kv_buffer": self.kv_buffer[:, index], "indexer_k_buffer": self.indexer_k_buffer[:, index]}

    def load_index_kv_buffer(self, index, load_tensor_dict):
        self.kv_buffer[:, index].copy_(load_tensor_dict["kv_buffer"])
        self.indexer_k_buffer[:, index].copy_(load_tensor_dict["indexer_k_buffer"])

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
