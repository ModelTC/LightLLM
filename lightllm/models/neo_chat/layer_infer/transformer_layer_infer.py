import os
import torch
from functools import partial
from typing import Tuple
from lightllm.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.models.neo_chat_moe.infer_struct import NeoChatInferStateInfo
from lightllm.models.llama.triton_kernel.token_attention_nopad_att1 import token_att_fwd
from lightllm.models.qwen3.layer_infer.transformer_layer_infer import Qwen3TransformerLayerInfer
from lightllm.models.neo_chat.layer_weights.transformer_layer_weight import NeoChatTransformerLayerWeight
from lightllm.distributed import all_reduce
import torch.distributed as dist
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.common.basemodel.attention.base_att import AttControl
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class NeoChatTransformerLayerInfer(Qwen3TransformerLayerInfer):
    def __init__(self, data_type, network_config):
        super().__init__(data_type, network_config)
        return

    def _get_qkv(self, input, infer_state: NeoChatInferStateInfo, layer_weight: NeoChatTransformerLayerWeight):
        input = input.view(-1, self.embed_dim_)

        qkv = layer_weight.qkv_proj.mm(input)
        q, cache_kv = qkv.split(
            [self.tp_q_head_num_ * self.head_dim_, (self.tp_k_head_num_ + self.tp_v_head_num_) * self.head_dim_], dim=-1
        )
        q = q.view(q.shape[0], self.tp_q_head_num_, self.head_dim_)
        q_t, q_hw = q.chunk(2, dim=-1)

        cache_kv = cache_kv.view(-1, (self.tp_k_head_num_ + self.tp_v_head_num_), self.head_dim_)
        k = cache_kv[:, : self.tp_k_head_num_, :]
        v = cache_kv[:, self.tp_k_head_num_ :, :]
        k_t, k_hw = k.chunk(2, dim=-1)

        q_t_2d = q_t.reshape(q.shape[0], -1)
        q_hw_2d = q_hw.reshape(q.shape[0], -1)
        k_t_2d = k_t.reshape(k.shape[0], -1)
        k_hw_2d = k_hw.reshape(k.shape[0], -1)

        layer_weight.qk_norm_weight_(q_t_2d, k_t_2d, eps=self.eps_)
        layer_weight.qk_hw_norm_weight_(q_hw_2d, k_hw_2d, eps=self.eps_)

        q_t = q_t_2d.view(q.shape[0], self.tp_q_head_num_, self.head_dim_ // 2)
        q_hw = q_hw_2d.view(q.shape[0], self.tp_q_head_num_, self.head_dim_ // 2)
        q_h, q_w = q_hw.chunk(2, dim=-1)

        k_t = k_t_2d.view(k.shape[0], self.tp_k_head_num_, self.head_dim_ // 2)
        k_hw = k_hw_2d.view(k.shape[0], self.tp_k_head_num_, self.head_dim_ // 2)
        k_h, k_w = k_hw.chunk(2, dim=-1)

        rotary_emb_fwd(
            q_t,
            k_t,
            infer_state.position_cos,
            infer_state.position_sin,
        )
        rotary_emb_fwd(
            q_h,
            k_h,
            infer_state.position_cos_h,
            infer_state.position_sin_h,
        )
        rotary_emb_fwd(
            q_w,
            k_w,
            infer_state.position_cos_w,
            infer_state.position_sin_w,
        )

        q = torch.cat([q_t, q_h, q_w], dim=-1)
        q = q.reshape(q.shape[0], -1)

        k = torch.cat([k_t, k_h, k_w], dim=-1)
        cache_kv = torch.cat([k, v], dim=1)
        return q, cache_kv
