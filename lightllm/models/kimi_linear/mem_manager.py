import torch

from lightllm.common.kv_cache_mem_manager.deepseek2_mem_manager import Deepseek2MemoryManager
from lightllm.common.kv_cache_mem_manager.qwen3next_mem_manager import (
    Qwen3NextLinearAttPageHelper,
    Qwen3NextMemManager,
)


class KimiLinearMemManager(Qwen3NextMemManager):
    def get_att_input_params(self, layer_index: int):
        layer_index = self.linear_config.get_full_attention_layer_index(layer_index)
        return self.kv_buffer[layer_index]

    def get_cell_size(self):
        return self.head_num * self.head_dim * self.layer_num * torch._utils._element_size(self.dtype)

    def _init_buffers(self, size, dtype, head_num, head_dim, layer_num):
        self.kv_buffer = torch.empty((layer_num, size + 1, head_num, head_dim), dtype=dtype, device="cuda")
        self._init_linear_att_buffers()

    def alloc_paged_kv_move_buffer(self, page_num, page_size):
        kv_move_buffer = Deepseek2MemoryManager.alloc_paged_kv_move_buffer(self, page_num, page_size)
        Qwen3NextLinearAttPageHelper(self).assert_page_size()
        return kv_move_buffer

    def write_mem_to_page_kv_move_buffer(self, *args, page_kind="kv", **kwargs):
        if page_kind == "kv":
            return Deepseek2MemoryManager.write_mem_to_page_kv_move_buffer(self, *args, page_kind=page_kind, **kwargs)
        return super().write_mem_to_page_kv_move_buffer(*args, page_kind=page_kind, **kwargs)

    def read_page_kv_move_buffer_to_mem(self, *args, page_kind="kv", **kwargs):
        if page_kind == "kv":
            return Deepseek2MemoryManager.read_page_kv_move_buffer_to_mem(self, *args, page_kind=page_kind, **kwargs)
        return super().read_page_kv_move_buffer_to_mem(*args, page_kind=page_kind, **kwargs)
