import os
import json
from lightllm.models.registry import ModelRegistry
from lightllm.models.qwen3next.model import Qwen3NextTpPartModel
from lightllm.models.qwen3_5.layer_weights.transformer_layer_weight import (
    Qwen35NextFullAttentionTransformerLayerWeight,
    Qwen35NextGatedDeltaNetTransformerLayerWeight,
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

    # Override to use multimodal pre-layer for vision processing
    pre_layer_infer_class = Qwen3VLMultimodalPreLayerInfer

    # Override to use multimodal pre/post weights (includes vision weights)
    pre_and_post_weight_class = Qwen3VLPreAndPostLayerWeight

    # Override to use Qwen3.5 infer state with mrope support
    infer_state_class = Qwen35InferStateInfo

    def __init__(self, kvargs):
        """
        Initialize Qwen3.5 model.

        Args:
            kvargs: Dictionary containing:
                - weight_dir: Path to model weights
                - max_total_token_num: Maximum total tokens
                - Additional model configuration
        """
        super().__init__(kvargs)
        logger.info("Initialized Qwen3.5 multimodal model")

    def _init_config(self):
        """
        Load and parse Qwen3.5 configuration.

        Qwen3.5 uses a nested config structure:
        {
            "model_type": "qwen3_5",
            "text_config": { ... },
            "vision_config": { ... }
        }

        This method extracts the text_config for the language model
        and stores vision_config for multimodal processing.
        """
        config_path = os.path.join(self.weight_dir_, "config.json")

        with open(config_path, "r") as json_file:
            all_config = json.load(json_file)

            # Extract text config for language model
            self.config = all_config["text_config"]

            # Store vision config for multimodal components
            self.vision_config = all_config.get("vision_config", None)

            if self.vision_config is None:
                logger.warning("No vision_config found in checkpoint. " "Multimodal features may not work correctly.")

        # Apply standard config repairs
        repair_config(self.config, same_names=["num_attention_heads", "n_head"])
        repair_config(self.config, same_names=["hidden_size", "n_embd", "n_embed"])
        repair_config(self.config, same_names=["num_hidden_layers", "n_layer"])

        # Qwen3.5 uses layer_types array instead of decoder_sparse_step for MoE placement
        # Set default for decoder_sparse_step (used by inherited Qwen3Next weight initialization)
        # Default to 1 meaning all layers with num_experts > 0 use MoE
        if "decoder_sparse_step" not in self.config:
            self.config["decoder_sparse_step"] = 1

        # Ensure mlp_only_layers exists (default to empty list)
        if "mlp_only_layers" not in self.config:
            self.config["mlp_only_layers"] = []

        # Qwen3.5 MoE uses moe_intermediate_size instead of intermediate_size
        # Set intermediate_size for compatibility with base layer weight classes
        if "intermediate_size" not in self.config:
            if "moe_intermediate_size" in self.config:
                self.config["intermediate_size"] = self.config["moe_intermediate_size"]
            else:
                # Default fallback: 4x hidden_size (common in transformer architectures)
                self.config["intermediate_size"] = self.config.get("hidden_size", 4096) * 4

        # Qwen3.5 stores RoPE config under text_config.rope_parameters.
        # Qwen3Next/llama infer path expects flattened keys like rope_theta and
        # partial_rotary_factor on the main config dict.
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

    def _init_weights(self):
        self.pre_post_weight = self.pre_and_post_weight_class(self.data_type, network_config=self.config)
        num_full_attention_layers = self.config["full_attention_interval"]
        self.trans_layers_weight = [
            (
                Qwen35NextFullAttentionTransformerLayerWeight(
                    i,
                    self.data_type,
                    network_config=self.config,
                    quant_cfg=self.quant_cfg,
                )
                if (i + 1) % num_full_attention_layers == 0
                else Qwen35NextGatedDeltaNetTransformerLayerWeight(
                    i,
                    self.data_type,
                    network_config=self.config,
                    quant_cfg=self.quant_cfg,
                )
            )
            for i in range(self.config["n_layer"])
        ]

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
