from lightllm.models.gpt_oss.layer_infer.transformer_layer_infer import GptOssTransformerLayerInfer
from lightllm.models.gpt_oss.layer_weights.transformer_layer_weight import GptOssTransformerLayerWeight
from lightllm.models.llama.model import LlamaTpPartModel
from lightllm.models.registry import ModelRegistry
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.log_utils import init_logger
from lightllm.common.basemodel.attention import get_prefill_att_backend_class, get_decode_att_backend_class
from lightllm.common.basemodel.attention import BaseAttBackend

logger = init_logger(__name__)


@ModelRegistry("gpt_oss")
class GptOssTpPartModel(LlamaTpPartModel):
    # weight class
    transformer_weight_class = GptOssTransformerLayerWeight

    # infer class
    transformer_layer_infer_class = GptOssTransformerLayerInfer

    def __init__(self, kvargs):
        super().__init__(kvargs)

    def _init_att_backend(self):
        self.prefill_att_backend: BaseAttBackend = get_prefill_att_backend_class(index=0, priority_list=["fa3"])(
            model=self
        )
        self.decode_att_backend: BaseAttBackend = get_decode_att_backend_class(index=0, priority_list=["fa3"])(
            model=self
        )
