from lightllm.models.deepseek_v4_dspark.layer_weights.pre_and_post_layer_weight import (
    DeepseekV4DSparkPreAndPostLayerWeight,
)


class DeepseekV41DSparkPreAndPostLayerWeight(DeepseekV4DSparkPreAndPostLayerWeight):
    has_hc_head = False
    markov_w1_name = "embed"
    markov_w2_name = "head"
