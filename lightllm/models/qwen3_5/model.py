import os
import json
from lightllm.models.registry import ModelRegistry
from lightllm.models.qwen3next.model import Qwen3NextTpPartModel
from lightllm.models.qwen3_5.layer_weights.transformer_layer_weight import (
    Qwen35TransformerLayerWeight,
)
from lightllm.models.qwen3_vl.model import QWen3VLTokenizer
from lightllm.models.qwen3_vl.layer_infer.pre_layer_infer import (
    Qwen3VLMultimodalPreLayerInfer,
)
from lightllm.models.qwen3_5.layer_weights.pre_and_post_layer_weight import (
    Qwen35PreAndPostLayerWeight,
)
from lightllm.models.qwen3_5.layer_infer.transformer_layer_infer import (
    Qwen35TransformerLayerInfer,
)
from lightllm.models.qwen3_5.infer_struct import Qwen35InferStateInfo
from lightllm.common.build_utils import repair_config
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class QWen3_5Tokenizer(QWen3VLTokenizer):
    """
    Tokenizer for Qwen3.5 multimodal model.

    Inherits all multimodal tokenization logic from Qwen3VL,
    including image and video token handling.
    """

    def __init__(self, tokenizer=None, image_processor=None, **kwargs):
        super().__init__(tokenizer, image_processor, **kwargs)


@ModelRegistry(["qwen3_5"], is_multimodal=True)
class Qwen3_5TpPartModel(Qwen3NextTpPartModel):
    """
    Qwen3.5 Multimodal Model (Dense Variant)

    This model combines:
    - Hybrid attention from Qwen3Next (Gated Delta Networks + Full Attention)
    - Multimodal capabilities from Qwen3VL (image/video processing)
    - Dense MLP layers (non-MoE)

    Architecture:
        - Every Nth layer uses full attention (config: full_attention_interval)
        - Other layers use linear attention (Gated Delta Networks)
        - Vision encoder processes images/videos before text model
        - Multimodal embeddings merged with text embeddings
    """

    transformer_weight_class = Qwen35TransformerLayerWeight
    pre_and_post_weight_class = Qwen35PreAndPostLayerWeight

    pre_layer_infer_class = Qwen3VLMultimodalPreLayerInfer
    transformer_layer_infer_class = Qwen35TransformerLayerInfer

    infer_state_class = Qwen35InferStateInfo

    def _init_config(self):
        config_path = os.path.join(self.weight_dir_, "config.json")

        with open(config_path, "r") as json_file:
            all_config = json.load(json_file)

            self.config = all_config["text_config"]
            self.vision_config = all_config.get("vision_config", None)

            if self.vision_config is None:
                logger.warning("No vision_config found in checkpoint. " "Multimodal features may not work correctly.")

        repair_config(self.config, same_names=["num_attention_heads", "n_head"])
        repair_config(self.config, same_names=["hidden_size", "n_embd", "n_embed"])
        repair_config(self.config, same_names=["num_hidden_layers", "n_layer"])

        # Rope parameters may live in three different places across HF / fine-tune
        # checkpoints: text_config.rope_parameters.<k>, text_config.<k>, or the root
        # all_config.<k>. Resolve each field by searching in that order and promote
        # into self.config. Missing rope parameters silently default to full rotary
        # in downstream code, which scrambles attention for models with
        # partial_rotary_factor != 1.0 — log the resolved value loudly.
        text_rope = self.config.get("rope_parameters") if isinstance(self.config.get("rope_parameters"), dict) else {}
        root_rope = all_config.get("rope_parameters") if isinstance(all_config.get("rope_parameters"), dict) else {}

        def _resolve(key):
            if key in text_rope:
                return text_rope[key], "text_config.rope_parameters"
            if key in self.config:
                return self.config[key], "text_config"
            if key in root_rope:
                return root_rope[key], "rope_parameters (root)"
            if key in all_config:
                return all_config[key], "root"
            return None, None

        for key in ("rope_theta", "partial_rotary_factor"):
            value, source = _resolve(key)
            if value is not None:
                self.config[key] = value
                logger.info(f"qwen3_5 _init_config: {key}={value} resolved from {source}")
            else:
                logger.warning(f"qwen3_5 _init_config: {key} NOT FOUND in config — downstream will use default")

        if "rope_scaling" not in self.config:
            if text_rope:
                self.config["rope_scaling"] = text_rope
            elif root_rope:
                self.config["rope_scaling"] = root_rope

        # MoE routing parameters - set defaults for Qwen3.5 compatibility
        if "norm_topk_prob" not in self.config:
            self.config["norm_topk_prob"] = True

        if self.finetune_config:
            self.config["vocab_size"] = self.finetune_config.vocab_size
        self.num_kv_heads = max(self.config["num_key_value_heads"] // self.tp_world_size_, 1)
