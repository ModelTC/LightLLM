from lightllm.models.deepseek_v4.layer_infer.pre_layer_infer import (
    DeepseekV4PreLayerInfer,
)
from lightllm.models.deepseek_v4_dspark.layer_weights.pre_and_post_layer_weight import (
    DeepseekV4DSparkPreAndPostLayerWeight,
)


class DeepseekV4DSparkPreLayerInfer(DeepseekV4PreLayerInfer):
    """Project selected target-layer hiddens for draft KV commits."""

    def __init__(self, network_config):
        super().__init__(network_config)
        self.eps_ = network_config["rms_norm_eps"]

    def context_forward(
        self,
        input_ids,
        infer_state,
        layer_weight: DeepseekV4DSparkPreAndPostLayerWeight,
    ):
        target_hidden = layer_weight.main_proj_weight_.mm(
            infer_state.mtp_draft_input_hiddens,
            use_custom_tensor_mananger=False,
        )
        return layer_weight.main_norm_weight_(
            input=target_hidden,
            eps=self.eps_,
            alloc_func=self.alloc_tensor,
        )
