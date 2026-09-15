from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig


class Qwen3_5DraftConfig(Qwen3_5TextConfig):
    """Older DFlash/DSpark checkpoints label their flat text backbone qwen3_5."""

    model_type = "qwen3_5"
    base_config_key = ""

    def get_text_config(self, **kwargs):
        return self
