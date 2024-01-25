import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np

from lightllm.utils.infer_utils import mark_cost_time
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.models.chatglm2.layer_weights.transformer_layer_weight import ChatGLM2TransformerLayerWeight

from lightllm.models.chatglm2.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.common.basemodel.triton_kernel.destindex_copy_kv import destindex_copy_kv, destindex_copy_quantize_kv
from lightllm.models.llama.triton_kernel.rmsnorm import rmsnorm_forward


class ChatGLM2TransformerLayerInfer(LlamaTransformerLayerInfer):
    """ """

    def __init__(self, layer_num, tp_rank, world_size, network_config, mode=[]):
        super().__init__(layer_num, tp_rank, world_size, network_config, mode)
        return

    def swiglu(self, x):
        x = torch.chunk(x, 2, dim=-1)
        return torch.nn.functional.silu(x[0]) * x[1]

    def _get_qkv(
        self, input_emb, cache_kv, infer_state: LlamaInferStateInfo, layer_weight: ChatGLM2TransformerLayerWeight
    ):
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
        return q, cache_kv

    def _ffn(self, input, infer_state: LlamaInferStateInfo, layer_weight: ChatGLM2TransformerLayerWeight):

        ffn1_out = torch.mm(input.view(-1, self.embed_dim_), layer_weight.gate_up_proj)
        act_out = self.swiglu(ffn1_out)
        ffn1_out = None
        ffn2_out = torch.mm(act_out, layer_weight.down_proj)
        return ffn2_out
