import torch
from .mem_manager import MemoryManager
from typing import Any
from lightllm.utils.log_utils import init_logger
from .operator import MlaMemOperator, FP8IndexedMlaMemOperator


logger = init_logger(__name__)


class MlaMemoryManager(MemoryManager):

    kv_is_replicated = True
    operator_class = MlaMemOperator

    def get_att_input_params(self, layer_index: int) -> Any:
        layer_index = self.get_kv_layer_index(layer_index)
        kv = self.kv_buffer[layer_index]
        return kv

    def get_cell_size(self):
        return self.head_num * self.head_dim * self.layer_num * torch._utils._element_size(self.dtype)

    def _init_buffers(self, size, dtype, head_num, head_dim, layer_num):
        assert size % self.page_size == 0, f"KV cache size {size} must be a multiple of page_size {self.page_size}"
        self.kv_buffer = torch.empty((layer_num, size + self.page_size, head_num, head_dim), dtype=dtype, device="cuda")

    def get_paged_kv_move_buffer_shape(self, page_num, page_size):
        # DeepSeek MLA's single compressed KV latent is replicated across TP ranks,
        # so the transfer buffer only needs one copy and does not multiply head_num by TP size.
        return (page_num, page_size, self.layer_num, self.head_num, self.head_dim)


class IndexedMlaMemoryManager(MlaMemoryManager):

    operator_class = MlaMemOperator

    def __init__(self, size, dtype, head_num, head_dim, layer_num, always_copy=False, mem_fraction=0.9):
        assert dtype in [torch.bfloat16, torch.float16]
        # 因为V3.2 使用了NSA 稀疏的缘故，所以其head_dim 会比原始的kv 多 128 + 4 = 132 个字节 (128 fp8 + 4byte float32 scale)，
        # 但是为了让整个数组具备16字节对齐，满足一些算子的约束，修改为添加 128 + 16 = 144 个字节, 这 144个字节中，后面132个字节用于
        # 存储真实数据，剩下12个，浪费了，只是占位。
        # 所以在子类中定制为其pad上，对外使用的接口，需要进行重载区别。
        super().__init__(size, dtype, head_num, head_dim + (144 // 2), layer_num, always_copy, mem_fraction)

    def get_att_input_params(self, layer_index: int) -> Any:
        layer_index = self.get_kv_layer_index(layer_index)
        kv = self.kv_buffer[layer_index][:, :, : (self.head_dim - (144 // 2))]
        return kv

    def get_indexer_k_buffer(self, layer_index: int) -> torch.Tensor:
        layer_index = self.get_kv_layer_index(layer_index)
        return self.kv_buffer[layer_index].view(dtype=torch.uint8)[:, :, -132:]


class FP8IndexedMlaMemoryManager(MlaMemoryManager):

    operator_class = FP8IndexedMlaMemOperator

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

    def get_att_input_params(self, layer_index: int) -> Any:
        layer_index = self.get_kv_layer_index(layer_index)
        return self.kv_buffer[layer_index][:, :, : self.flashmla_bytes_per_token]

    def get_indexer_k_buffer(self, layer_index: int) -> torch.Tensor:
        layer_index = self.get_kv_layer_index(layer_index)
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
        from lightllm.common.basemodel.triton_kernel.kv_copy.prefill_compact_kv_flashmla_fp8 import (
            get_prefill_kv_cache_and_remap_indices_triton,
        )

        return get_prefill_kv_cache_and_remap_indices_triton(
            packed_kv=packed_kv,
            topk_mem_indices=topk_indices,
            prefill_mem_index=prefill_mem_index,
            prefill_cache_kv=prefill_cache_kv,
            prefill_dtype=self.prefill_dtype,
        )
