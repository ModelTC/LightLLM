import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np

from lightllm.utils.infer_utils import mark_cost_time
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.models.llama2.layer_infer.transformer_layer_infer import Llama2TransformerLayerInfer
from lightllm.models.chatglm2.layer_weights.transformer_layer_weight import ChatGLM2TransformerLayerWeight

from lightllm.models.chatglm2.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.common.basemodel.triton_kernel.destindex_copy_kv import destindex_copy_kv, destindex_copy_quantize_kv
from lightllm.models.llama.triton_kernel.rmsnorm import rmsnorm_forward


class ChatGLM2TransformerLayerInfer(Llama2TransformerLayerInfer):
    """
    """
    def __init__(self, layer_num, tp_rank, world_size, network_config, mode=[]):
        super().__init__(layer_num, tp_rank, world_size, network_config, mode)
        return

    def swiglu(self, x):
        x = torch.chunk(x, 2, dim=-1)
        return torch.nn.functional.silu(x[0]) * x[1]

    def _get_qkv(self, input_emb, cache_k, cache_v, infer_state: LlamaInferStateInfo, layer_weight:ChatGLM2TransformerLayerWeight):        
        q = torch.addmm(layer_weight.q_bias_, input_emb.view(-1, self.embed_dim_), layer_weight.q_weight_, beta=1.0, alpha=1.0)
        rotary_emb_fwd(q.view(-1, self.tp_q_head_num_, self.head_dim_), infer_state.position_cos, infer_state.position_sin)
        torch.addmm(layer_weight.k_bias_, input_emb.view(-1, self.embed_dim_), layer_weight.k_weight_, beta=1.0, alpha=1.0,
                    out=cache_k.view(-1, self.tp_k_head_num_ * self.head_dim_))
        rotary_emb_fwd(cache_k, infer_state.position_cos, infer_state.position_sin)
        torch.addmm(layer_weight.v_bias_, input_emb.view(-1, self.embed_dim_), layer_weight.v_weight_, beta=1.0, alpha=1.0,
                    out=cache_v.view(-1, self.tp_v_head_num_ * self.head_dim_))
        return q

    def _ffn(self, input, infer_state: LlamaInferStateInfo, layer_weight: ChatGLM2TransformerLayerWeight):

        ffn1_out = torch.mm(input.view(-1, self.embed_dim_), layer_weight.ffn_1_weight_)
        act_out = self.swiglu(ffn1_out)
        ffn1_out = None
        ffn2_out = torch.mm(act_out, layer_weight.ffn_2_weight_)
        return ffn2_out