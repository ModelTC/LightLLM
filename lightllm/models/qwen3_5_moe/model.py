from lightllm.models.registry import ModelRegistry
from lightllm.models.qwen3_5.model import Qwen3_5TpPartModel
from lightllm.utils.log_utils import init_logger
from lightllm.distributed.communication_op import dist_group_manager

logger = init_logger(__name__)


@ModelRegistry("qwen3_5_moe", is_multimodal=True)
class Qwen3_5MOETpPartModel(Qwen3_5TpPartModel):
    def __init__(self, kvargs):
        super().__init__(kvargs)
        logger.info("Initialized Qwen3.5-MoE multimodal model with expert routing")

    def _init_custom(self):
        super()._init_custom()
        # Initialize DeepEP group for MoE models with num_experts
        if "num_experts" in self.config and self.config["num_experts"] > 0:
            dist_group_manager.new_deepep_group(self.config["num_experts"], self.config["hidden_size"])
