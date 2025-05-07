from lightllm.models.registry import ModelRegistry, is_reward_model
from lightllm.models.qwen2_reward.layer_infer.post_layer_infer import Qwen2RewardPostLayerInfer
from lightllm.models.qwen2_reward.layer_weights.pre_and_post_layer_weight import Qwen2RewardPreAndPostLayerWeight
from lightllm.models.qwen2.model import Qwen2TpPartModel


@ModelRegistry("qwen2", condition=is_reward_model())
class Qwen2RewardTpPartModel(Qwen2TpPartModel):

    pre_and_post_weight_class = Qwen2RewardPreAndPostLayerWeight
    post_layer_infer_class = Qwen2RewardPostLayerInfer

    def __init__(self, kvargs):
        super().__init__(kvargs)
