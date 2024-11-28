import torch

from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.models.llama.layer_infer.post_layer_infer import LlamaPostLayerInfer
from lightllm.models.qwen2_reward.layer_weights.pre_and_post_layer_weight import Qwen2RewardPreAndPostLayerWeight
from einops import rearrange


class Qwen2RewardPostLayerInfer(LlamaPostLayerInfer):
    def token_forward(
        self, input_embdings, infer_state: LlamaInferStateInfo, layer_weight: Qwen2RewardPreAndPostLayerWeight
    ):
        last_input, token_num = self._slice_get_last_input(input_embdings, infer_state)

        input_embdings = None
        last_input = self._norm(last_input, infer_state, layer_weight)

        last_input = torch.addmm(layer_weight.score_up_bias, last_input, layer_weight.score_up_weight)
        last_input = torch.nn.functional.relu(last_input)
        score = torch.addmm(layer_weight.score_down_bias, last_input, layer_weight.score_down_weight)

        return score
