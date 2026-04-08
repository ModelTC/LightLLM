import torch
from .mem_manager import MemoryManager
from .kv_buffer.quant_kv_buffer import PPLInt4QuantKvBuffer


class PPLINT4KVMemoryManager(MemoryManager):
    def __init__(self, size, dtype, head_num, head_dim, layer_num, always_copy=True, mem_fraction=0.9):
        self.kv_dtype = torch.int8
        self.group_quant_size = 8
        super().__init__(size, dtype, head_num, head_dim, layer_num, always_copy=always_copy, mem_fraction=mem_fraction)

    def get_cell_size(self):
        return 2 * self.head_num * self.head_dim // 2 * self.layer_num * torch._utils._element_size(
            self.kv_dtype
        ) + 2 * self.head_num * self.head_dim // self.group_quant_size * self.layer_num * torch._utils._element_size(
            self.dtype
        )

    def _init_buffers(self, size, dtype, head_num, head_dim, layer_num):
        self.kv_buffer = PPLInt4QuantKvBuffer(
            torch.empty((layer_num, size + 1, 2 * head_num, head_dim // 2), dtype=torch.int8, device="cuda"),
            scale_buffer=torch.empty(
                (layer_num, size + 1, 2 * head_num, head_dim // self.group_quant_size), dtype=dtype, device="cuda"
            ),
            head_num=head_num,
            quant_group_size=self.group_quant_size,
        )
