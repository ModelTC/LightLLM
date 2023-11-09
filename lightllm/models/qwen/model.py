import os
import json
import torch
import math

from .layer_infer.transformer_layer_infer import QwenTransformerLayerInfer
from .layer_weights.pre_and_post_layer_weight import QwenPreAndPostLayerWeight
from .layer_weights.transformer_layer_weight import QwenTransformerLayerWeight
from .infer_struct import QwenInferStateInfo

from lightllm.models.llama.model import LlamaTpPartModel
from lightllm.common.build_utils import repair_config


class QWenTpPartModel(LlamaTpPartModel):
    # weight class
    pre_and_post_weight_class = QwenPreAndPostLayerWeight
    transformer_weight_class = QwenTransformerLayerWeight

    # infer class
    transformer_layer_infer_class = QwenTransformerLayerInfer

    # infer state class
    infer_state_class = QwenInferStateInfo

    def __init__(self, kvargs):
        super().__init__(kvargs)
    
    def _init_config(self):
        super()._init_config()
        # rename key
        # repair_config()
        repair_config(self.config, same_names=["ffn_hidden_size", "intermediate_size"])
        repair_config(self.config, same_names=["rms_norm_eps", "layer_norm_epsilon"])
        return 
    
    def _init_custom(self):
        """
        init qwen dynamic_ntk and logn_attn
        """
        if self.config.get("use_dynamic_ntk", False) and self.config.get("use_logn_attn", False):
            self._init_qwen_dynamic_ntk()
            self._init_qwen_logn_attn()
        else:
            super()._init_custom()
            self.logn_tensor = None
        return
    
    def _init_qwen_dynamic_ntk(self):
        total_seq_len_supported = self.config.get("max_position_embeddings", 8 * 1024)
        seq_len = self.config.get("seq_length", 2048)

        cur_kv_seq_len = total_seq_len_supported
        context_value = math.log(cur_kv_seq_len / seq_len, 2) + 1
        ntk_alpha = 2 ** math.ceil(context_value) - 1
        ntk_alpha = max(ntk_alpha, 1)

        base = self.config.get("rotary_emb_base", 10000)
        base = base * ntk_alpha ** (self.head_dim_ / (self.head_dim_ - 2))
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, self.head_dim_, 2, device="cpu", dtype=torch.float32)
                / self.head_dim_
            )
        )

        t = torch.arange(total_seq_len_supported + 64 * 1024, device="cpu", dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)

        self._cos_cached = torch.cos(freqs).to(torch.float16).cuda()
        self._sin_cached = torch.sin(freqs).to(torch.float16).cuda()
        return
    
    def _init_qwen_logn_attn(self):
        total_seq_len_supported = self.config.get("max_position_embeddings", 8 * 1024)
        seq_len = self.config.get("seq_length", 2048)
        logn_list = [
            math.log(i, seq_len) if i > seq_len else 1
            for i in range(1, total_seq_len_supported + 64 * 1024 + 1)
        ]
        self.logn_tensor = torch.tensor(logn_list).cuda()
        return
