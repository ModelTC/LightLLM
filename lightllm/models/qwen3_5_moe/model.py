from lightllm.models.registry import ModelRegistry
from lightllm.models.qwen3_5.model import Qwen3_5TpPartModel
from lightllm.utils.log_utils import init_logger
from lightllm.distributed.communication_op import dist_group_manager

logger = init_logger(__name__)


@ModelRegistry("qwen3_5_moe", is_multimodal=True)
class Qwen3_5MOETpPartModel(Qwen3_5TpPartModel):
    """
    Qwen3.5-MoE Multimodal Model (Mixture of Experts Variant)

    Extends Qwen3.5 with sparse expert routing:
    - Same hybrid attention architecture as Qwen3.5
    - MoE layers replace dense MLP layers
    - Expert routing handled by inherited MoE infrastructure

    This model combines:
    - Hybrid attention from Qwen3Next (Gated Delta Networks + Full Attention)
    - Multimodal capabilities from Qwen3VL (image/video processing)
    - MoE sparse routing for efficient scaling
    """

    def __init__(self, kvargs):
        """
        Initialize Qwen3.5-MoE model.

        Args:
            kvargs: Dictionary containing:
                - weight_dir: Path to model weights
                - max_total_token_num: Maximum total tokens
                - Additional model configuration
        """
        super().__init__(kvargs)
        logger.info("Initialized Qwen3.5-MoE multimodal model with expert routing")

    def _init_custom(self):
        """
        Initialize MoE-specific components.

        Sets up DeepEP communication group for expert parallelism
        when the model has experts configured.
        """
        super()._init_custom()
        # Initialize DeepEP group for MoE models with num_experts
        if "num_experts" in self.config and self.config["num_experts"] > 0:
            dist_group_manager.new_deepep_group(self.config["num_experts"], self.config["hidden_size"])
