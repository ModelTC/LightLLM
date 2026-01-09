import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np
from typing import Tuple
from functools import partial
import triton

from lightllm.models.gemma_2b.layer_weights.transformer_layer_weight import Gemma_2bTransformerLayerWeight
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.models.gemma_2b.triton_kernel.gelu_and_mul import gelu_and_mul_fwd

from lightllm.models.llama.infer_struct import LlamaInferStateInfo


class Gemma_2bTransformerLayerInfer(LlamaTransformerLayerInfer):
    """ """

    def __init__(self, layer_num, network_config):
        super().__init__(layer_num, network_config)
        self.tp_k_head_num_ = network_config["num_key_value_heads"]  # [SYM] always == 1
        self.tp_v_head_num_ = network_config["num_key_value_heads"]
        return

    def _ffn(
        self, input, infer_state: LlamaInferStateInfo, layer_weight: Gemma_2bTransformerLayerWeight
    ) -> torch.Tensor:
        up_gate_out = layer_weight.gate_up_proj.mm(input.view(-1, self.embed_dim_))
        ffn1_out = self.alloc_tensor((input.size(0), up_gate_out.size(1) // 2), input.dtype)
        gelu_and_mul_fwd(up_gate_out, ffn1_out)
        input = None
        up_gate_out = None
        ffn2_out = layer_weight.down_proj.mm(
            ffn1_out,
        )
        ffn1_out = None
        return ffn2_out
