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
from lightllm.models.qwen3_vl.layer_weights.pre_and_post_layer_weight import (
    Qwen3VLPreAndPostLayerWeight,
)
from lightllm.models.qwen3_5.layer_infer.transformer_layer_infer import (
    Qwen35FullAttentionTransformerLayerInfer,
    Qwen35GatedDeltaNetTransformerLayerInfer,
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

    pre_layer_infer_class = Qwen3VLMultimodalPreLayerInfer
    transformer_weight_class = Qwen35TransformerLayerWeight
    pre_and_post_weight_class = Qwen3VLPreAndPostLayerWeight

    infer_state_class = Qwen35InferStateInfo

    def _init_config(self):
        config_path = os.path.join(self.weight_dir_, "config.json")

        with open(config_path, "r") as json_file:
            all_config = json.load(json_file)

            self.config = all_config["text_config"]
            self.vision_config = all_config.get("vision_config", None)

            if self.vision_config is None:
                logger.warning("No vision_config found in checkpoint. " "Multimodal features may not work correctly.")

        # Apply standard config repairs
        repair_config(self.config, same_names=["num_attention_heads", "n_head"])
        repair_config(self.config, same_names=["hidden_size", "n_embd", "n_embed"])
        repair_config(self.config, same_names=["num_hidden_layers", "n_layer"])

        # Qwen3.5 stores RoPE config under text_config.rope_parameters.
        rope_parameters = self.config.get("rope_parameters")
        if isinstance(rope_parameters, dict):
            if "rope_theta" in rope_parameters and "rope_theta" not in self.config:
                self.config["rope_theta"] = rope_parameters["rope_theta"]
            if "partial_rotary_factor" in rope_parameters and "partial_rotary_factor" not in self.config:
                self.config["partial_rotary_factor"] = rope_parameters["partial_rotary_factor"]
            # Preserve the richer RoPE metadata in the expected field when absent.
            if "rope_scaling" not in self.config:
                self.config["rope_scaling"] = rope_parameters

        # MoE routing parameters - set defaults for Qwen3.5 compatibility
        if "norm_topk_prob" not in self.config:
            self.config["norm_topk_prob"] = True  # Standard default for MoE models

        # Handle fine-tuning config if present
        if self.finetune_config:
            self.config["vocab_size"] = self.finetune_config.vocab_size

        # Calculate num_kv_heads for KV cache memory management
        # Required by parent class _init_mem_manager() in Qwen3NextTpPartModel
        self.num_kv_heads = max(self.config["num_key_value_heads"] // self.tp_world_size_, 1)

    def _init_infer_layer(self):
        """
        Initialize inference layers for Qwen3.5 multimodal model.

        Uses mrope-enabled transformer layers to properly handle image/video
        tokens with 3D position encoding (temporal, height, width).

        This overrides the parent class to use Qwen35* layer classes instead
        of Qwen3Next* layer classes.
        """
        self.pre_infer = self.pre_layer_infer_class(network_config=self.config)
        self.post_infer = self.post_layer_infer_class(network_config=self.config)
        num_full_attention_layers = self.config["full_attention_interval"]

        self.layers_infer = [
            (
                Qwen35FullAttentionTransformerLayerInfer(i, network_config=self.config)
                if (i + 1) % num_full_attention_layers == 0
                else Qwen35GatedDeltaNetTransformerLayerInfer(i, network_config=self.config)
            )
            for i in range(self.config["n_layer"])
        ]
