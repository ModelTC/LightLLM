import torch

from lightllm.models.deepseek_v4.infer_struct import DeepseekV4InferStateInfo
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
        self.is_last_layer = layer_num == final_layer
        assert (
            self.compress_ratio == 0
        ), "DeepSeek-V4 DSpark draft layers must be SWA-only"

    def context_forward(
        self,
        input_embdings: torch.Tensor,
        infer_state: DeepseekV4InferStateInfo,
        layer_weight: DeepseekV4DSparkTransformerLayerWeight,
    ) -> torch.Tensor:
        """Write target hidden rows into this stage without running draft attention/FFN."""
        full_input = self._tpsp_allgather(input=input_embdings, infer_state=infer_state)
        qkv = layer_weight.wq_a_wkv_.mm(full_input, use_custom_tensor_mananger=False)
        infer_state.mem_manager.pack_mla_kv_to_cache_fused_norm_rope(
            layer_index=self.layer_num_,
            mem_index=infer_state.mem_index,
            kv=qkv[:, -self.head_dim_ :],
            kv_weight=layer_weight.kv_norm_.weight,
            eps=self.eps_,
            freqs_cis=self.freqs_cis,
            positions=infer_state.position_ids,
        )
        return input_embdings
