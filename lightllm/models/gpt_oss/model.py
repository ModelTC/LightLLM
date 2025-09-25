import torch
import numpy as np

from lightllm.models.gpt_oss.layer_infer.transformer_layer_infer import GptOssTransformerLayerInfer
from lightllm.models.gpt_oss.layer_weights.transformer_layer_weight import GptOssTransformerLayerWeight
from lightllm.models.llama.layer_infer.post_layer_infer import LlamaPostLayerInfer
from lightllm.models.llama.layer_infer.pre_layer_infer import LlamaPreLayerInfer
from lightllm.models.llama.layer_weights.pre_and_post_layer_weight import LlamaPreAndPostLayerWeight
from lightllm.models.llama.model import LlamaTpPartModel
from lightllm.models.registry import ModelRegistry
from lightllm.common.basemodel.basemodel import TpPartBaseModel
from lightllm.common.mem_manager import MemoryManager

from lightllm.models.vit.infer_struct import LlamaInferStateInfo
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


@ModelRegistry("gpt_oss")
class GptOssTpPartModel(LlamaTpPartModel):
    # weight class
    transformer_weight_class = GptOssTransformerLayerWeight

    # infer class
    transformer_layer_infer_class = GptOssTransformerLayerInfer

    def __init__(self, kvargs):
        super().__init__(kvargs)
        assert get_env_start_args().enable_fa3, "For now GPT-OSS type model only support flashattention-3"
