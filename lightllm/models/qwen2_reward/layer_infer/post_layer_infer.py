import torch

from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.models.llama.layer_infer.post_layer_infer import LlamaPostLayerInfer
from lightllm.models.qwen2_reward.layer_weights.pre_and_post_layer_weight import Qwen2RewardPreAndPostLayerWeight


class Qwen2RewardPostLayerInfer(LlamaPostLayerInfer):
    def token_forward(
        self, input_embdings, infer_state: LlamaInferStateInfo, layer_weight: Qwen2RewardPreAndPostLayerWeight
    ):
        last_input, token_num = self._slice_get_last_input(input_embdings, infer_state)

        input_embdings = None
        last_input = self._norm(last_input, infer_state, layer_weight)

        last_input = layer_weight.score_up_weight_.mm(last_input)
        last_input = torch.nn.functional.relu(last_input)
        score = layer_weight.score_down_weight_.mm(last_input)

        return score
