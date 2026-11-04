from lightllm.models.qwen3.layer_infer.transformer_layer_infer import Qwen3TransformerLayerInfer
from lightllm.models.qwen3.layer_weights.transformer_layer_weight import Qwen3TransformerLayerWeight
from lightllm.models.qwen2.model import Qwen2TpPartModel, Qwen2ModelAdapter
from lightllm.utils.log_utils import init_logger


logger = init_logger(__name__)


class Qwen3ModelAdapter(Qwen2ModelAdapter):
    transformer_weight_class = Qwen3TransformerLayerWeight

    transformer_layer_infer_class = Qwen3TransformerLayerInfer


class Qwen3TpPartModel(Qwen2TpPartModel):
    model_adapter_class = Qwen3ModelAdapter
    transformer_weight_class = Qwen3ModelAdapter.transformer_weight_class
    transformer_layer_infer_class = Qwen3ModelAdapter.transformer_layer_infer_class

    def __init__(self, kvargs):
        super().__init__(kvargs)
        return
