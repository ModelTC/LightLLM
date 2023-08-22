import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np

from lightllm.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.models.internlm.layer_weights.transformer_layer_weight import InternlmTransformerLayerWeight
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.models.llama.infer_struct import LlamaInferStateInfo


class InternlmTransformerLayerInfer(LlamaTransformerLayerInfer):

    def __init__(self, layer_num, tp_rank, world_size, network_config, mode=""):
        super().__init__(layer_num, tp_rank, world_size, network_config, mode)
        return
    
    def _get_qkv(self, input, cache_k, cache_v, infer_state:LlamaInferStateInfo, layer_weight:InternlmTransformerLayerWeight)->torch.Tensor:
        q = torch.addmm(layer_weight.q_bias_, input.view(-1, self.embed_dim_), layer_weight.q_weight_, beta=1.0, alpha=1.0)
        rotary_emb_fwd(q.view(-1, self.tp_q_head_num_, self.head_dim_), infer_state.position_cos, infer_state.position_sin)
        torch.addmm(layer_weight.k_bias_, input.view(-1, self.embed_dim_), layer_weight.k_weight_, beta=1.0,
                    alpha=1.0, out=cache_k.view(-1, self.tp_k_head_num_ * self.head_dim_))
        rotary_emb_fwd(cache_k, infer_state.position_cos, infer_state.position_sin)
        torch.addmm(layer_weight.v_bias_, input.view(-1, self.embed_dim_), layer_weight.v_weight_, beta=1.0,
                    alpha=1.0, out=cache_v.view(-1, self.tp_v_head_num_ * self.head_dim_))
        return q

    def _get_o(self, input, infer_state:LlamaInferStateInfo, layer_weight:InternlmTransformerLayerWeight)->torch.Tensor:
        o_tensor = torch.addmm(layer_weight.o_bias_, input.view(-1, self.tp_o_head_num_ * self.head_dim_), layer_weight.o_weight_, beta=1.0 / self.world_size_)
        return o_tensor