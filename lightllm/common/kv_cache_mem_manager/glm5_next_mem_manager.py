import torch

from .mla_mem_manager import MlaMemoryManager
from .operator import NormalMemOperator
from lightllm.common.basemodel.triton_kernel.destindex_copy_kv import destindex_copy_kv


class Glm5NextMemOperator(NormalMemOperator):
    def copy_kv_to_mem_manager(self, layer_index, mem_index, kv):
        output = self.mem_manager.get_att_input_params(layer_index)[:, :, : kv.shape[-1]]
        destindex_copy_kv(kv, mem_index, output)


class Glm5NextMemManager(MlaMemoryManager):
    """Packed MLA and indexer KV storage, replicated across TP ranks."""

    operator_class = Glm5NextMemOperator

    def __init__(
        self,
        size,
        dtype,
        head_num,
        head_dim,
        layer_num,
        always_copy=False,
        mem_fraction=0.9,
        mla_head_dim=512,
    ):
        assert head_num == 1
        self.mla_head_dim = mla_head_dim
        super().__init__(size, dtype, head_num, head_dim, layer_num, always_copy, mem_fraction)

    def _layer_buffer(self, layer_index):
        return self.kv_buffer[self.get_kv_layer_index(layer_index)]

    def get_att_input_params(self, layer_index):
        return self._layer_buffer(layer_index)[:, :, : self.mla_head_dim]

    def get_indexer_k_buffer(self, layer_index):
        return self._layer_buffer(layer_index).view(torch.uint8)[:, :, -132:]
