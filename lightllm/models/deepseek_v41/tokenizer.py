import os

from lightllm.models.deepseek_v4.tokenizer import DeepSeekV4Tokenizer


class DeepSeekV41Tokenizer(DeepSeekV4Tokenizer):
    supports_content_blocks = True
    image_token_alignment = 1

    def __init__(self, tokenizer, model_dir, model_config):
        self.vision_config = model_config.get("vision_config") or {}
        config = {**model_config["text_config"], **model_config}
        config.setdefault("vision_n_layers", self.vision_config.get("num_hidden_layers", 0))
        super().__init__(tokenizer, model_dir, config)

    def apply_chat_template(self, *args, **kwargs):
        # Accept legacy OpenAI effort names at the model's tokenizer boundary.
        effort = kwargs.get("reasoning_effort")
        if effort == "medium":
            kwargs["reasoning_effort"] = "low"
        elif effort in ("minimal", "xhigh"):
            kwargs["reasoning_effort"] = "high"
        return super().apply_chat_template(*args, **kwargs)

    def get_image_token_length(self, img):
        from .image_processor import get_image_grid

        cfg = self.vision_config
        _, _, _, _, token_num = get_image_grid(
            img.image_h,
            img.image_w,
            cfg["patch_size"],
            cfg["downsample_ratio"],
            cfg["max_image_tokens"],
            cfg["min_pixels"],
            cfg.get("max_wh_ratio"),
        )
        return token_num

    def _get_encoding_path(self):
        return os.path.join(self.model_dir, "encoding", "encoding.py")
