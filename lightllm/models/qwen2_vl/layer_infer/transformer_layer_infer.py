import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np
from functools import partial

from lightllm.models.qwen2_vl.triton_kernel.mrope import mrope_triton
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer


class Qwen2VLTransformerLayerInfer(LlamaTransformerLayerInfer):
    def __init__(self, layer_num, network_config, mode=[]):
        super().__init__(layer_num, network_config, mode)
        self.mrope_section = network_config["rope_scaling"]["mrope_section"]
        axis_map = []
        for i, n in enumerate(self.mrope_section * 2):
            axis_map += [i % 3] * n
        self.axis_map = torch.tensor(axis_map, dtype=torch.int32, device="cuda")

    def _get_qkv(self, input, infer_state, layer_weight):
        q = layer_weight.q_proj.mm(input)
        cache_kv = self._pre_cache_kv(infer_state=infer_state, layer_weight=layer_weight)
        cache_kv = layer_weight.kv_proj.mm(
            input, out=cache_kv.view(-1, (self.tp_k_head_num_ + self.tp_v_head_num_) * self.head_dim_)
        ).view(-1, (self.tp_k_head_num_ + self.tp_v_head_num_), self.head_dim_)
        seq_len, _ = q.shape
        q = q.view(1, seq_len, -1, self.head_dim_).transpose(1, 2)
        self.axis_map = self.axis_map.to(q.device)
        k = cache_kv[:, : self.tp_k_head_num_, :].view(1, seq_len, -1, self.head_dim_).transpose(1, 2)
        new_q, new_k = mrope_triton(q, k, infer_state.position_cos, infer_state.position_sin, self.axis_map)
        new_q = new_q.transpose(1, 2).reshape(1, seq_len, -1)
        cache_kv[:, : self.tp_k_head_num_, :] = new_k.squeeze(0).permute(1, 0, 2)

        return new_q, cache_kv
