import os
import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np
import triton
from typing import Tuple
from lightllm.models.qwen3_vl_moe.layer_infer.transformer_layer_infer import Qwen3VLMOETransformerLayerInfer
from lightllm.models.qwen3_moe.layer_infer.transformer_layer_infer import Qwen3MOETransformerLayerInfer
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.models.llama.triton_kernel.silu_and_mul import silu_and_mul_fwd
from functools import partial
from lightllm.utils.log_utils import init_logger
from lightllm.utils.dist_utils import get_global_world_size
from lightllm.distributed.communication_op import all_gather_into_tensor, reduce_scatter_tensor

logger = init_logger(__name__)


class Qwen3OmniMOETransformerLayerInfer(Qwen3VLMOETransformerLayerInfer):
    def __init__(self, layer_num, network_config):
        self.layer_num_ = network_config["num_hidden_layers"]
        super().__init__(layer_num, network_config)
        self.head_dim_ = network_config["head_dim"]
        self.mrope_section = torch.tensor(
            network_config["rope_scaling"]["mrope_section"], dtype=torch.int32, device="cuda"
        )
        return
