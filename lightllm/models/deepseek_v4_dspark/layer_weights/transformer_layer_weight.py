from lightllm.models.deepseek_v4.layer_weights.transformer_layer_weight import (
    DeepseekV4TransformerLayerWeight,
)


class DeepseekV4DSparkTransformerLayerWeight(DeepseekV4TransformerLayerWeight):
    """A DSpark stage whose cache layer id is global but checkpoint prefix is local."""

    def _parse_config(self):
        super()._parse_config()
        stage_id = self.layer_num_ - self.network_config_["n_layer"]
        assert 0 <= stage_id < self.network_config_["dspark_layer_num"]
        self.prefix = f"mtp.{stage_id}"
