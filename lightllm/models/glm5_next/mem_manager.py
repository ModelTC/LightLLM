# SPDX-License-Identifier: Apache-2.0

import torch

from lightllm.common.kv_cache_mem_manager.operator import LinearAttMemOperator
from lightllm.common.kv_cache_mem_manager.qwen3next_mem_manager import (
    Qwen3NextMemManager,
)


class Glm5NextMemOperator(LinearAttMemOperator):
    def copy_kv_to_mem_manager(self, layer_index, mem_index, kv):
        layer_index = self.linear_config.get_full_att_kv_layer_index(layer_index)
        from lightllm.common.basemodel.triton_kernel.destindex_copy_kv import (
            destindex_copy_kv,
        )

        output = self.mem_manager.kv_buffer[layer_index][:, :, : self.mem_manager.mla_head_dim]
        destindex_copy_kv(kv, mem_index, output)


class Glm5NextMemManager(Qwen3NextMemManager):
    """Packed sparse-MLA KV, DSA index keys, and KDA recurrent states."""

    operator_class = Glm5NextMemOperator
    indexer_padding_bytes = 144
    indexer_payload_bytes = 132

    def __init__(self, *args, **kwargs):
        self.mla_head_dim = kwargs.get("head_dim", args[3] if len(args) > 3 else None)
        super().__init__(*args, **kwargs)

    def get_cell_size(self):
        bytes_per_token = self.mla_head_dim * self.dtype.itemsize + self.indexer_padding_bytes
        return bytes_per_token * self.layer_num

    def _init_buffers(self, size, dtype, head_num, head_dim, layer_num):
        assert head_num == 1 and dtype in (torch.bfloat16, torch.float16)
        padding_elements = self.indexer_padding_bytes // dtype.itemsize
        self.kv_buffer = torch.empty(
            (layer_num, size + 1, head_num, head_dim + padding_elements),
            dtype=dtype,
            device="cuda",
        )
        self._init_linear_att_buffers()

    def get_att_input_params(self, layer_index):
        packed_index = self.linear_config.get_full_att_kv_layer_index(layer_index)
        return self.kv_buffer[packed_index][:, :, : self.mla_head_dim]

    def get_indexer_k_buffer(self, layer_index):
        packed_index = self.linear_config.get_full_att_kv_layer_index(layer_index)
        return self.kv_buffer[packed_index].view(torch.uint8)[:, :, -self.indexer_payload_bytes :]
