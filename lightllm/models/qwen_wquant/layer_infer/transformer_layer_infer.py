import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np

from lightllm.models.llama_wquant.layer_infer.transformer_layer_infer import LlamaTransformerLayerInferWquant
from lightllm.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.models.qwen_wquant.layer_weights.transformer_layer_weight import QwenTransformerLayerWeightQuantized
from lightllm.models.qwen.infer_struct import QwenInferStateInfo

class QwenTransformerLayerInferWQuant(LlamaTransformerLayerInferWquant):
    """
    """

    def __init__(self, layer_num, tp_rank, world_size, network_config, mode=[]):
        super().__init__(layer_num, tp_rank, world_size, network_config, mode)
        self.inter_dim_ = network_config['intermediate_size'] // 2 # qwen 的 inter_dim 要 // 2
        return
    
    def _get_qkv(self, input, cache_k, cache_v, infer_state: QwenInferStateInfo, layer_weight: QwenTransformerLayerWeightQuantized):
        qkv_output = self._wquant_matmul_for_qkv(input.view(-1, self.embed_dim_), 
                                                    quant_weight_params=layer_weight.qkv_weight_,
                                                    is_prefill=infer_state.is_prefill,
                                                    bias=layer_weight.qkv_bias_)
        
        tp_k_head_dim = self.tp_k_head_num_ * self.head_dim_
        q = qkv_output[:, : -2 * tp_k_head_dim]
        k = qkv_output[:, -2 * tp_k_head_dim: -tp_k_head_dim]
        v = qkv_output[:, -tp_k_head_dim :]

        if infer_state.logn_values is not None:
            q.mul_(infer_state.logn_values.view(-1, 1))

        rotary_emb_fwd(q.view(-1, self.tp_q_head_num_, self.head_dim_), infer_state.position_cos, infer_state.position_sin)
        cache_k_ = k.view(-1, self.tp_k_head_num_, self.head_dim_)
        rotary_emb_fwd(cache_k_, infer_state.position_cos, infer_state.position_sin)
        cache_v_ = v.view(-1, self.tp_v_head_num_, self.head_dim_)
        return q, cache_k_, cache_v_

