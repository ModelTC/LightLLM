import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np
import triton

from lightllm.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.models.internlm_wquant.layer_weights.transformer_layer_weight import InternlmTransformerLayerWeightQuantized
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.models.llama_wquant.layer_infer.transformer_layer_infer import LlamaTransformerLayerInferWquant

class InternlmTransformerLayerInferWquant(LlamaTransformerLayerInferWquant):

    def __init__(self, layer_num, tp_rank, world_size, network_config, mode=[]):
        super().__init__(layer_num, tp_rank, world_size, network_config, mode)
        return

    def _get_qkv(self, input, cache_k, cache_v, infer_state: LlamaInferStateInfo, layer_weight: InternlmTransformerLayerWeightQuantized):
        qkv_output = self._wquant_matmul_for_qkv(input.view(-1, self.embed_dim_), 
                                                    quant_weight_params=layer_weight.qkv_weight_,
                                                    infer_state=infer_state)
        
        tp_k_head_dim = self.tp_k_head_num_ * self.head_dim_
        q = qkv_output[:, : -2 * tp_k_head_dim].add_(layer_weight.q_bias_)
        k = qkv_output[:, -2 * tp_k_head_dim: -tp_k_head_dim].add_(layer_weight.k_bias_)
        v = qkv_output[:, -tp_k_head_dim :].add_(layer_weight.v_bias_)

        rotary_emb_fwd(q.view(-1, self.tp_q_head_num_, self.head_dim_), infer_state.position_cos, infer_state.position_sin)
        cache_k_ = k.view(-1, self.tp_k_head_num_, self.head_dim_)
        rotary_emb_fwd(cache_k_, infer_state.position_cos, infer_state.position_sin)
        cache_v_ = v.view(-1, self.tp_v_head_num_, self.head_dim_)
        return q, cache_k_, cache_v_

    def _get_o(self, input, infer_state: LlamaInferStateInfo, layer_weight: InternlmTransformerLayerWeightQuantized) -> torch.Tensor:
        o_tensor = self._wquant_matmul_for_o(input, 
                                             quant_weight_params=layer_weight.o_weight_,
                                             infer_state=infer_state,
                                             bias=layer_weight.o_bias_ / self.world_size_)
        return o_tensor