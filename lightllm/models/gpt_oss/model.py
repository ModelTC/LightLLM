from lightllm.models.gpt_oss.layer_infer.transformer_layer_infer import GptOssTransformerLayerInfer
from lightllm.models.gpt_oss.layer_weights.transformer_layer_weight import GptOssTransformerLayerWeight
from lightllm.models.llama.model import LlamaTpPartModel
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.log_utils import init_logger
from lightllm.common.basemodel.attention import PREFILL, DECODE, resolve_att_backend_class
from lightllm.common.basemodel.attention import AttBackendSlot, MAIN_SLOT

logger = init_logger(__name__)


class GptOssTpPartModel(LlamaTpPartModel):
    # weight class
    transformer_weight_class = GptOssTransformerLayerWeight

    # infer class
    transformer_layer_infer_class = GptOssTransformerLayerInfer

    def __init__(self, kvargs):
        super().__init__(kvargs)

    def _init_att_backends(self):
        # Attention sinks are only implemented by FA3, so explicit flags are restricted as well.
        return {
            MAIN_SLOT: AttBackendSlot.build(
                self,
                prefill_class=resolve_att_backend_class("dense", PREFILL, allowed=("fa3",)),
                decode_class=resolve_att_backend_class("dense", DECODE, allowed=("fa3",)),
            )
        }
