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
        assert self.compress_ratio == 0, "DeepSeek-V4 DSpark draft layers must be SWA-only"

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
        infer_state.mem_manager.pack_mla_kv_to_cache_fused_norm_rope(
            layer_index=self.layer_num_,
            swa_slots=infer_state.dsv4_swa_write_slots,
            kv=infer_state.context_kv[self.stage_id],
            kv_weight=layer_weight.kv_norm_.weight,
            eps=self.eps_,
            freqs_cis=self.freqs_cis,
            positions=infer_state.position_ids,
        )
        return input_embdings
