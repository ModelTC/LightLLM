import torch

from lightllm.models.deepseek_v4_dspark.infer_struct import DeepseekV4DSparkInferStateInfo
from lightllm.models.deepseek_v4.layer_infer.transformer_layer_infer import (
    DeepseekV4TransformerLayerInfer,
)
from lightllm.models.deepseek_v4_dspark.layer_weights.transformer_layer_weight import (
    DeepseekV4DSparkTransformerLayerWeight,
)


class DeepseekV4DSparkTransformerLayerInfer(DeepseekV4TransformerLayerInfer):
    """Full DSpark stage with a KV-only target-hidden commit primitive."""

    def __init__(self, layer_num, network_config):
        super().__init__(layer_num, network_config)
        final_layer = network_config["n_layer"] + network_config["dspark_layer_num"] - 1
        self.stage_id = layer_num - network_config["n_layer"]
        self.is_last_layer = layer_num == final_layer

    def context_forward(
        self,
        input_embdings: torch.Tensor,
        infer_state: DeepseekV4DSparkInferStateInfo,
        layer_weight: DeepseekV4DSparkTransformerLayerWeight,
    ) -> torch.Tensor:
        """Write target hidden rows into this stage without running draft attention/FFN."""
        if self.stage_id == 0:
            all_kv = self.context_wkv_weight.mm(input_embdings, use_custom_tensor_mananger=False)
            infer_state.context_kv = all_kv.split(self.head_dim_, dim=-1)
        self._norm_rope_and_cache_kv(infer_state.context_kv[self.stage_id], infer_state, layer_weight)
        return input_embdings
