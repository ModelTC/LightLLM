import torch

from lightllm.common.kv_cache_mem_manager.operator import LinearAttMemOperator
from lightllm.common.kv_cache_mem_manager.qwen3next_mem_manager import Qwen3NextMemManager
from lightllm.common.basemodel.triton_kernel.destindex_copy_kv import destindex_copy_kv


class Glm5NextMemOperator(LinearAttMemOperator):
    def copy_kv_to_mem_manager(self, layer_index, mem_index, kv):
        output = self.mem_manager.get_att_input_params(layer_index)[:, :, : kv.shape[-1]]
        destindex_copy_kv(kv, mem_index, output)


class Glm5NextMemManager(Qwen3NextMemManager):
    """One packed token buffer; KDA uses the standard big/small page pools."""

    operator_class = Glm5NextMemOperator

    def __init__(self, *args, mla_head_dim=512, index_head_dim=128, **kwargs):
        self.mla_head_dim = mla_head_dim
        self.index_head_dim = index_head_dim
        super().__init__(*args, **kwargs)

    def get_cell_size(self):
        return self.head_dim * self.dtype.itemsize * self.layer_num

    def _init_buffers(self, size, dtype, head_num, head_dim, layer_num):
        assert head_num == 1
        self.kv_buffer = torch.empty((layer_num, size + 1, 1, head_dim), dtype=dtype, device="cuda")
        self.kv_buffer[:, :, :, self.mla_head_dim : self.mla_head_dim + 64].zero_()
        self._init_linear_att_buffers()

    def _layer_buffer(self, layer_index):
        return self.kv_buffer[self.linear_config.get_full_att_kv_layer_index(layer_index)]

    def get_att_input_params(self, layer_index):
        return self._layer_buffer(layer_index)[:, :, : self.mla_head_dim + 64]

    def get_indexer_raw_buffer(self, layer_index):
        start = self.mla_head_dim + 64
        return self._layer_buffer(layer_index)[:, :, start : start + 2 * self.index_head_dim]

    def get_indexer_k_buffer(self, layer_index):
        return self._layer_buffer(layer_index).view(torch.uint8)[:, :, -132:]
