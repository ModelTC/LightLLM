import torch
import torch.functional as F
import numpy as np
from lightllm.models.bloom.layer_weights.pre_and_post_layer_weight import BloomPreAndPostLayerWeight
from lightllm.distributed.communication_op import all_gather
from lightllm.models.llama.layer_infer.post_layer_infer import LlamaPostLayerInfer
from lightllm.common.build_utils import repair_config


class BloomPostLayerInfer(LlamaPostLayerInfer):
    """ """

    def __init__(self, network_config):
        repair_config(config=network_config, same_names=["layer_norm_epsilon", "rms_norm_eps"])
        super().__init__(network_config)
        return

    def _norm(self, input, infer_state, layer_weight: BloomPreAndPostLayerWeight) -> torch.Tensor:
        return layer_weight.final_norm_weight_.layernorm_forward(
            input=input, eps=self.eps_, alloc_func=self.alloc_tensor
        )
