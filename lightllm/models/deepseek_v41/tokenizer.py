import importlib.util
import os

from lightllm.models.deepseek_v4.tokenizer import DeepSeekV4Tokenizer


class DeepSeekV41Tokenizer(DeepSeekV4Tokenizer):
    supported_reasoning_efforts = ("low", "high", "max", None, *range(1, 101))

    def __init__(self, tokenizer, model_dir, model_config):
        self.vision_config = model_config.get("vision_config") or {}
        config = {**model_config["text_config"], **model_config, "vision_row_major": True}
        config.setdefault("vision_n_layers", self.vision_config.get("num_hidden_layers", 0))
        super().__init__(tokenizer, model_dir, config)

    def get_image_token_length(self, img):
        from lightllm.models.deepseek_v4.image_processor import get_image_grid

        cfg = self.vision_config
        _, _, _, _, token_num = get_image_grid(
            img.image_h,
            img.image_w,
            cfg["patch_size"],
            cfg["downsample_ratio"],
            cfg["max_image_tokens"],
            cfg["min_pixels"],
            cfg.get("max_wh_ratio"),
            row_major=True,
        )
        return token_num

    def _get_encoding_module(self):
        if self._encoding_module is None:
            path = os.path.join(self.model_dir, "encoding", "encoding.py")
            spec = importlib.util.spec_from_file_location("lightllm_deepseek_v41_encoding", path)
            self._encoding_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self._encoding_module)
        return self._encoding_module
