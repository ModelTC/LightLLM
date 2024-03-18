from typing import Tuple

import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np
from functools import partial

from lightllm.common.basemodel.triton_kernel.quantize_gemm_int8 import matmul_quantize_int8
from lightllm.common.basemodel.triton_kernel.dequantize_gemm_int8 import matmul_dequantize_int8
from lightllm.common.basemodel.triton_kernel.dequantize_gemm_int4 import (
    matmul_dequantize_int4_s1,
    matmul_dequantize_int4_s2,
    matmul_dequantize_int4_gptq,
)
from lightllm.models.starcoder_wquant.layer_weights.transformer_layer_weight import (
    StarcoderTransformerLayerWeightQuantized,
)
from lightllm.utils.infer_utils import mark_cost_time
from lightllm.models.starcoder.infer_struct import StarcoderInferStateInfo
from lightllm.common.basemodel import TransformerLayerInferWeightQuantTpl
from lightllm.models.bloom.layer_infer.transformer_layer_infer import BloomTransformerLayerInfer
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.models.llama_wquant.layer_infer.transformer_layer_infer import LlamaTransformerLayerInferWquant


class StarcoderTransformerLayerInferWQuant(TransformerLayerInferWeightQuantTpl):
    """ """

    def __init__(self, layer_num, tp_rank, world_size, network_config, mode=[]):
        super().__init__(layer_num, tp_rank, world_size, network_config, mode)
        self.eps_ = network_config["layer_norm_epsilon"]
        self.tp_q_head_num_ = network_config["num_attention_heads"] // self.world_size_
        self.tp_k_head_num_ = 1
        self.tp_v_head_num_ = 1
        self.tp_o_head_num_ = self.tp_q_head_num_
        self.head_dim_ = network_config["n_embed"] // network_config["num_attention_heads"]
        self.embed_dim_ = network_config["n_embed"]
        self._bind_func()
        return

    def _bind_func(self):
        self._att_norm = partial(BloomTransformerLayerInfer._att_norm, self)
        self._ffn_norm = partial(BloomTransformerLayerInfer._ffn_norm, self)

        LlamaTransformerLayerInferWquant._bind_matmul(self)
        LlamaTransformerLayerInfer._bind_attention(self)
        return

    def _get_qkv(
        self,
        input,
        cache_kv,
        infer_state: StarcoderInferStateInfo,
        layer_weight: StarcoderTransformerLayerWeightQuantized,
    ) -> torch.Tensor:
        q = self._wquant_matmul_for_qkv(
            input.view(-1, self.embed_dim_), layer_weight.q_weight_, infer_state=infer_state, bias=layer_weight.q_bias_
        )
        cache_kv = self._wquant_matmul_for_qkv(
            input.view(-1, self.embed_dim_),
            layer_weight.kv_weight_,
            infer_state=infer_state,
            bias=layer_weight.kv_bias_,
        ).view(-1, self.tp_k_head_num_ + self.tp_v_head_num_, self.head_dim_)
        return q, cache_kv

    def _get_o(
        self, input, infer_state: StarcoderInferStateInfo, layer_weight: StarcoderTransformerLayerWeightQuantized
    ) -> torch.Tensor:
        o_output = self._wquant_matmul_for_o(
            input.view(-1, self.embed_dim_), layer_weight.o_weight_, infer_state=infer_state, bias=layer_weight.o_bias_
        )
        return o_output

    def _ffn(
        self, input, infer_state: StarcoderInferStateInfo, layer_weight: StarcoderTransformerLayerWeightQuantized
    ) -> torch.Tensor:
        ffn1_out = self._wquant_matmul_for_ffn_up(
            input.view(-1, self.embed_dim_),
            layer_weight.ffn_1_weight_,
            infer_state=infer_state,
            bias=layer_weight.ffn_1_bias_,
        )
        input = None
        gelu_out = torch.nn.functional.gelu(ffn1_out, approximate="tanh")
        ffn1_out = None
        ffn2_out = self._wquant_matmul_for_ffn_down(
            gelu_out, layer_weight.ffn_2_weight_, infer_state=infer_state, bias=layer_weight.ffn_2_bias_
        )
        gelu_out = None
        return ffn2_out
