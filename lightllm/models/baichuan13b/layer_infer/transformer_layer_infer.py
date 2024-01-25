import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np
from functools import partial

from lightllm.models.baichuan13b.layer_weights.transformer_layer_weight import BaiChuan13bTransformerLayerWeight
from lightllm.models.bloom.layer_infer.transformer_layer_infer import BloomTransformerLayerInfer
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer


class Baichuan13bTransformerLayerInfer(LlamaTransformerLayerInfer):
    """ """

    def __init__(self, layer_num, tp_rank, world_size, network_config, mode):
        super().__init__(layer_num, tp_rank, world_size, network_config, mode)
        self._bind_func()
        return

    def _bind_func(self):
        """
        baichuan13b only support normal mode.
        """
        self._context_attention_kernel = partial(BloomTransformerLayerInfer._context_attention_kernel, self)
        self._token_attention_kernel = partial(BloomTransformerLayerInfer._token_attention_kernel, self)
        return

    def _get_qkv(self, input, cache_kv, infer_state, layer_weight: BaiChuan13bTransformerLayerWeight) -> torch.Tensor:
        q = torch.mm(input.view(-1, self.embed_dim_), layer_weight.q_weight_)
        torch.mm(
            input.view(-1, self.embed_dim_),
            layer_weight.kv_weight_,
            out=cache_kv.view(-1, (self.tp_k_head_num_ + self.tp_v_head_num_) * self.head_dim_),
        )
        return q, cache_kv
