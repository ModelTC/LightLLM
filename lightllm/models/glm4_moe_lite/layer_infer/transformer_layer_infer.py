import os
import torch
import torch.distributed as dist
import triton
from functools import partial
from lightllm.models.deepseek2.layer_infer.transformer_layer_infer import Deepseek2TransformerLayerInfer
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.distributed.communication_op import reduce_scatter_tensor


class Glm4MoeLiteTransformerLayerInfer(Deepseek2TransformerLayerInfer):
    def __init__(self, layer_num, network_config):
        self._glm4_layer_num = layer_num
        self._glm4_first_k_dense = network_config.get("first_k_dense_replace", 0)
        self._glm4_has_routed_experts = network_config.get("n_routed_experts") is not None
        super().__init__(layer_num, network_config)

    @property
    def is_moe(self):
        return self._glm4_has_routed_experts and self._glm4_layer_num >= self._glm4_first_k_dense

    @is_moe.setter
    def is_moe(self, value):
        pass
