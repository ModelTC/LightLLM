import torch
import torch.distributed as dist
import numpy as np

from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.models.llama.layer_infer.post_layer_infer import LlamaPostLayerInfer
from lightllm.models.llama.layer_weights.pre_and_post_layer_weight import LlamaPreAndPostLayerWeight
from einops import rearrange


class Internlm2RewardPostLayerInfer(LlamaPostLayerInfer):
    def token_forward(self, input_embdings, infer_state: LlamaInferStateInfo, layer_weight: LlamaPreAndPostLayerWeight):
        last_input, token_num = self._slice_get_last_input(input_embdings, infer_state)

        input_embdings = None
        last_input = self._norm(last_input, infer_state, layer_weight)
        score = torch.mm(last_input, layer_weight.lm_head_weight_)

        return score
