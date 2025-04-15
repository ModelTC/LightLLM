import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np

from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from functools import partial


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    mrope_section = mrope_section * 2
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(unsqueeze_dim)
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


class Qwen2VLTransformerLayerInfer(LlamaTransformerLayerInfer):
    def __init__(self, layer_num, network_config, mode=[]):
        super().__init__(layer_num, network_config, mode)
        self.mrope_section = network_config["rope_scaling"]["mrope_section"]

    def _get_qkv(self, input, cache_kv, infer_state, layer_weight):
        q = layer_weight.q_proj.mm(input)
        cache_kv = layer_weight.kv_proj.mm(
            input, out=cache_kv.view(-1, (self.tp_k_head_num_ + self.tp_v_head_num_) * self.head_dim_)
        ).view(-1, (self.tp_k_head_num_ + self.tp_v_head_num_), self.head_dim_)
        seq_len, _ = q.shape
        q = q.view(1, seq_len, -1, self.head_dim_).transpose(1, 2)
        k = cache_kv[:, : self.tp_k_head_num_, :].view(1, seq_len, -1, self.head_dim_).transpose(1, 2)
        new_q, new_k = apply_multimodal_rotary_pos_emb(
            q, k, infer_state.position_cos, infer_state.position_sin, self.mrope_section
        )
        new_q = new_q.transpose(1, 2).reshape(1, seq_len, -1)
        cache_kv[:, : self.tp_k_head_num_, :] = new_k.squeeze(0).permute(1, 0, 2)

        return new_q, cache_kv
