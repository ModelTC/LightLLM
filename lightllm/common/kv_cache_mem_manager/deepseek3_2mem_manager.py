import torch
from typing import Any
from lightllm.common.kv_cache_mem_manager.deepseek2_mem_manager import Deepseek2MemoryManager


class Deepseek3_2MemoryManager(Deepseek2MemoryManager):
    def __init__(self, size, dtype, head_num, head_dim, layer_num, always_copy=False, mem_fraction=0.9):
        assert dtype in [torch.bfloat16, torch.float16]
        # 因为V3.2 使用了NSA 稀疏的缘故，所以其head_dim 会比原始的kv 多 128 + 4 = 132 个字节 (128 fp8 + 4byte float32 scale)，
        # 但是为了让整个数组具备16字节对齐，满足一些算子的约束，修改为添加 128 + 16 = 144 个字节, 这 144个字节中，后面132个字节用于
        # 存储真实数据，剩下12个，浪费了，只是占位。
        # 所以在子类中定制为其pad上，对外使用的接口，需要进行重载区别。
        super().__init__(size, dtype, head_num, head_dim + (144 // 2), layer_num, always_copy, mem_fraction)

    def copy_kv_to_mem_manager(self, layer_index: int, mem_index: torch.Tensor, kv: torch.Tensor):
        """
        将每一层生成的kv拷贝到mem manager对应mem_index 位置中
        """
        from ..basemodel.triton_kernel.kv_copy.mla_copy_kv import destindex_copy_kv

        rope_dim = 64
        kv_lora_rank = kv.shape[2] - rope_dim
        assert kv_lora_rank + rope_dim == self.kv_buffer.shape[-1] - (144 // 2)

        destindex_copy_kv(
            kv[:, :, :kv_lora_rank],
            kv[:, :, kv_lora_rank:],
            mem_index,
            self.kv_buffer[layer_index][:, :, :kv_lora_rank],
            self.kv_buffer[layer_index][:, :, kv_lora_rank : (kv_lora_rank + rope_dim)],
        )
        return

    def get_att_input_params(self, layer_index: int) -> Any:
        kv = self.kv_buffer[layer_index][:, :, : (self.head_dim - (144 // 2))]
        return kv

    def get_indexer_k_buffer(self, layer_index: int) -> torch.Tensor:
        return self.kv_buffer[layer_index].view(dtype=torch.uint8)[:, :, -132:]
