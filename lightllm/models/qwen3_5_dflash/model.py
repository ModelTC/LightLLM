from lightllm.models.llama.model import LlamaTpPartModel
from lightllm.models.qwen3_dflash.model import Qwen3DFlashModel
from lightllm.models.qwen3_5_dflash.layer_weights.pre_and_post_layer_weight import (
    Qwen35DFlashPreAndPostLayerWeight,
)


class Qwen3_5DFlashModel(Qwen3DFlashModel):
    """Adapter for a Qwen3 DFlash checkpoint paired with a Qwen3.5 target."""

    pre_and_post_weight_class = Qwen35DFlashPreAndPostLayerWeight

    @classmethod
    def normalize_model_config(cls, model_config: dict[str, object]) -> dict[str, object]:
        """合并 draft 专有配置，再同步别名及 rotary 参数；启动与 CPU cache 使用同一结果。"""
        config = model_config.get("text_config", model_config).copy()
        dflash_config = config.get("dflash_config", {})
        config.update(dflash_config)
        config = super().normalize_model_config(config)

        rope_parameters = config["rope_parameters"]
        if "rope_theta" in rope_parameters and "rope_theta" not in config:
            config["rope_theta"] = rope_parameters["rope_theta"]
        if "partial_rotary_factor" in rope_parameters and "partial_rotary_factor" not in config:
            config["partial_rotary_factor"] = rope_parameters["partial_rotary_factor"]
        if "rope_scaling" not in config:
            config["rope_scaling"] = rope_parameters
        return config

    def _init_custom(self):
        # Draft and target use different rotary shapes, so the draft owns its rotary cache.
        LlamaTpPartModel._init_custom(self)
        self.block_size = self.config["block_size"]
        self.mask_token_id = self.config["mask_token_id"]

    def _init_weights(self, start_layer_index=None):
        super()._init_weights(start_layer_index=start_layer_index)
        self.pre_post_weight.wte_weight_ = self.main_model.pre_post_weight.wte_weight_
        self.pre_post_weight.lm_head_weight_ = self.main_model.pre_post_weight.lm_head_weight_
