from lightllm.models.llama.model import LlamaTpPartModel
from lightllm.models.qwen3_dspark.model import Qwen3DSparkModel


class Qwen3_5DSparkModel(Qwen3DSparkModel):
    """Adapter for the current Qwen3 DSpark checkpoint with a Qwen3.5 target."""

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

        # The draft is Qwen3-style and owns a 1D rotary cache. Released
        # checkpoints rotate the full head, while target-shaped custom
        # checkpoints can retain Qwen3.5's partial rotary layout.
        config["rope_scaling"] = rope_parameters
        config["partial_rotary_factor"] = rope_parameters.get("partial_rotary_factor", 1.0)
        return config

    def _init_custom(self):
        # Draft and target use different rotary shapes, so the draft owns its rotary cache.
        LlamaTpPartModel._init_custom(self)
        self.block_size = self.config["block_size"]
        self.mask_token_id = self.config["mask_token_id"]
