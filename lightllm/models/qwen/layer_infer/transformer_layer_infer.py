import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np
from typing import Tuple
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.models.qwen.layer_weights.transformer_layer_weight import QwenTransformerLayerWeight
from lightllm.models.qwen.infer_struct import QwenInferStateInfo


class QwenTransformerLayerInfer(LlamaTransformerLayerInfer):
    """ """

    def __init__(self, layer_num, network_config, mode=[]):
        super().__init__(layer_num, network_config, mode)
        return

    def _get_qkv(self, input_emb, infer_state: QwenInferStateInfo, layer_weight: QwenTransformerLayerWeight):
        qkv = layer_weight.qkv_proj.mm(input)
        q, cache_kv = qkv.split(
            [self.tp_q_head_num_ * self.head_dim_, (self.tp_k_head_num_ + self.tp_v_head_num_) * self.head_dim_], dim=-1
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

    def _tpsp_get_qkv(self, input, infer_state, layer_weight) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO
        raise Exception("not impl")
