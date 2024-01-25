import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np

from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.models.qwen.layer_weights.transformer_layer_weight import QwenTransformerLayerWeight
from lightllm.models.qwen.infer_struct import QwenInferStateInfo


class QwenTransformerLayerInfer(LlamaTransformerLayerInfer):
    """ """

    def __init__(self, layer_num, tp_rank, world_size, network_config, mode=[]):
        super().__init__(layer_num, tp_rank, world_size, network_config, mode)
        return

    def _get_qkv(self, input_emb, cache_kv, infer_state: QwenInferStateInfo, layer_weight: QwenTransformerLayerWeight):
        q = torch.addmm(
            layer_weight.q_bias_, input_emb.view(-1, self.embed_dim_), layer_weight.q_weight_, beta=1.0, alpha=1.0
        )
        torch.addmm(
            layer_weight.kv_bias_,
            input_emb.view(-1, self.embed_dim_),
            layer_weight.kv_weight_,
            beta=1.0,
            alpha=1.0,
            out=cache_kv.view(-1, (self.tp_k_head_num_ + self.tp_v_head_num_) * self.head_dim_),
        )
        rotary_emb_fwd(
            q.view(-1, self.tp_q_head_num_, self.head_dim_),
            cache_kv[:, 0 : self.tp_k_head_num_, :],
            infer_state.position_cos,
            infer_state.position_sin,
        )
        if infer_state.logn_values is not None:
            q.mul_(infer_state.logn_values.view(-1, 1))
        return q, cache_kv
