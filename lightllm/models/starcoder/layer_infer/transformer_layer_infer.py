import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np
from functools import partial

from lightllm.utils.infer_utils import mark_cost_time
from lightllm.models.starcoder.infer_struct import StarcoderInferStateInfo
from lightllm.models.bloom.layer_infer.transformer_layer_infer import BloomTransformerLayerInfer
from lightllm.models.starcoder.layer_weights.transformer_layer_weight import StarcoderTransformerLayerWeight
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer

class StarcoderTransformerLayerInfer(BloomTransformerLayerInfer):
    """
    """
    def __init__(self, layer_num, tp_rank, world_size, network_config, mode=[]):
        super().__init__(layer_num, tp_rank, world_size, network_config, mode)
        self.tp_k_head_num_ = 1
        self.tp_v_head_num_ = 1
        self._bind_func()
        return
    
    def _bind_func(self):
        LlamaTransformerLayerInfer._bind_attention(self)
        return