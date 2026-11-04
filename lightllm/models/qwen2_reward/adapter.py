from lightllm.models.qwen2_reward.layer_infer.post_layer_infer import Qwen2RewardPostLayerInfer
from lightllm.models.qwen2_reward.layer_weights.pre_and_post_layer_weight import Qwen2RewardPreAndPostLayerWeight
from lightllm.models.qwen2.adapter import Qwen2ModelAdapter


class Qwen2RewardModelAdapter(Qwen2ModelAdapter):
    pre_and_post_weight_class = Qwen2RewardPreAndPostLayerWeight
    post_layer_infer_class = Qwen2RewardPostLayerInfer
