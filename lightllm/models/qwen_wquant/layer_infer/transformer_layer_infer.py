import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np

from lightllm.models.llama_wquant.layer_infer.transformer_layer_infer import LlamaTransformerLayerInferWquant
from lightllm.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.models.qwen_wquant.layer_weights.transformer_layer_weight import QwenTransformerLayerWeightQuantized
from lightllm.models.qwen.infer_struct import QwenInferStateInfo


class QwenTransformerLayerInferWQuant(LlamaTransformerLayerInferWquant):
    """ """

    def __init__(self, layer_num, tp_rank, world_size, network_config, mode=[]):
        super().__init__(layer_num, tp_rank, world_size, network_config, mode)
        self.inter_dim_ = network_config["intermediate_size"] // 2  # qwen 的 inter_dim 要 // 2
        return

    def _get_qkv(
        self, input, cache_kv, infer_state: QwenInferStateInfo, layer_weight: QwenTransformerLayerWeightQuantized
    ):
        q = self._wquant_matmul_for_qkv(
            input.view(-1, self.embed_dim_),
            quant_weight_params=layer_weight.q_weight_,
            infer_state=infer_state,
            bias=layer_weight.q_bias_,
        )
        cache_kv = self._wquant_matmul_for_qkv(
            input.view(-1, self.embed_dim_),
            quant_weight_params=layer_weight.kv_weight_,
            infer_state=infer_state,
            bias=layer_weight.kv_bias_,
        ).view(-1, self.tp_k_head_num_ + self.tp_v_head_num_, self.head_dim_)
        rotary_emb_fwd(
            q.view(-1, self.tp_q_head_num_, self.head_dim_),
            cache_kv[:, 0 : self.tp_k_head_num_, :],
            infer_state.position_cos,
            infer_state.position_sin,
        )
        if infer_state.logn_values is not None:
            q.mul_(infer_state.logn_values.view(-1, 1))
        return q, cache_kv
