import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np

from lightllm.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.models.llama.triton_kernel.silu_and_mul import silu_and_mul_fwd
from lightllm.models.internlm_xcomposer.layer_weights.transformer_layer_weight import InternlmComposerTransformerLayerWeight
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.models.internlm_xcomposer.infer_struct import InternlmComposerInferStateInfo


class InternlmComposerTransformerLayerInfer(LlamaTransformerLayerInfer):
    def __init__(self, layer_num, tp_rank, world_size, network_config, mode=[]):
        super().__init__(layer_num, tp_rank, world_size, network_config, mode)
        return

    def _get_qkv(
        self, input, cache_kv, infer_state: InternlmComposerInferStateInfo, layer_weight: InternlmComposerTransformerLayerWeight
    ) -> torch.Tensor:
        im_mask = infer_state.im_mask
        has_img = infer_state.has_img
        q = torch.mm(
            input.view(-1, self.embed_dim_), layer_weight.q_weight_
        )
        torch.addmm(
            layer_weight.kv_bias_,
            input.view(-1, self.embed_dim_),
            layer_weight.kv_weight_,
            beta=1.0,
            alpha=1.0,
            out=cache_kv.view(-1, (self.tp_k_head_num_ + self.tp_v_head_num_) * self.head_dim_),
        )
        if has_img:
            input_part = input.view(-1, self.embed_dim_)[im_mask]
            q[im_mask] += torch.mm(
                torch.mm(input_part, layer_weight.qkv_loraA_weight_), 
                layer_weight.q_loraB_weight_
            )
            cache_kv[im_mask] += torch.mm(
                torch.mm(input_part, layer_weight.qkv_loraA_weight_), 
                layer_weight.kv_loraB_weight_
            ).view(-1, (self.tp_k_head_num_ + self.tp_v_head_num_), self.head_dim_)
        rotary_emb_fwd(
            q.view(-1, self.tp_q_head_num_, self.head_dim_),
            cache_kv[:, 0 : self.tp_k_head_num_, :],
            infer_state.position_cos,
            infer_state.position_sin,
        )
        return q, cache_kv

    def _get_o(
        self, input, infer_state: InternlmComposerInferStateInfo, layer_weight: InternlmComposerTransformerLayerWeight
    ) -> torch.Tensor:
        im_mask = infer_state.im_mask
        has_img = infer_state.has_img
        o_tensor = torch.mm(
            input.view(-1, self.tp_o_head_num_ * self.head_dim_),
            layer_weight.o_weight_,
        )
        if has_img:
            input_part = input.view(-1, self.tp_o_head_num_ * self.head_dim_)[im_mask]
            o_tensor[im_mask] += torch.mm(
                torch.mm(input_part, layer_weight.wo_loraA_weight_),
                layer_weight.wo_loraB_weight_
            )
        return o_tensor

    def _ffn(self, input, infer_state: InternlmComposerInferStateInfo, layer_weight: InternlmComposerTransformerLayerWeight) -> torch.Tensor:
        im_mask = infer_state.im_mask
        has_img = infer_state.has_img
        up_gate_out = torch.mm(input.view(-1, self.embed_dim_), layer_weight.gate_up_proj)
        if has_img:
            gate_dim = up_gate_out.shape[1] // 2
            input_part = input.view(-1, self.embed_dim_)[im_mask]
            up_gate_out[:, :gate_dim][im_mask] += torch.mm(
                torch.mm(input_part, layer_weight.gate_loraA_weight_),
                layer_weight.gate_loraB_weight_
            )
            up_gate_out[:, gate_dim:][im_mask] += torch.mm(
                torch.mm(input_part, layer_weight.up_loraA_weight_),
                layer_weight.up_loraB_weight_
            )
        ffn1_out = silu_and_mul_fwd(up_gate_out)
        input = None
        up_gate_out = None
        ffn2_out = torch.mm(ffn1_out, layer_weight.down_proj)
        if has_img:
            ffn2_out[im_mask] += torch.mm(
                torch.mm(ffn1_out[im_mask], layer_weight.down_loraA_weight_),
                layer_weight.down_loraB_weight_
            )
        ffn1_out = None
        return ffn2_out