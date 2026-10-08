from lightllm.models.qwen3_vl_moe.layer_infer.transformer_layer_infer import Qwen3VLMOETransformerLayerInfer
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class Qwen3OmniMOETransformerLayerInfer(Qwen3VLMOETransformerLayerInfer):
    def __init__(self, layer_num, network_config):
        super().__init__(layer_num, network_config)
        self.head_dim_ = network_config["head_dim"]
        return
