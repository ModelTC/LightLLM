import os
import json
from functools import partial

import torch

from lightllm.models.llama.layer_infer.pre_layer_infer import LlamaPreLayerInfer
from lightllm.models.llama.layer_infer.post_layer_infer import LlamaPostLayerInfer
from lightllm.models.llama.layer_weights.pre_and_post_layer_weight import LlamaPreAndPostLayerWeight
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.models.llama_quantized.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeightQuantized
from lightllm.models.llama_quantized.layer_infer.transformer_layer_infer import \
    LlamaTransformerLayerInferINT8, LlamaTransformerLayerInferINT4
from lightllm.common.mem_manager import MemoryManager
from lightllm.common.int8kv_mem_manager import INT8KVMemoryManager
from lightllm.models.llama.model import LlamaTpPartModel


class LlamaTpPartModelQuantized(LlamaTpPartModel):
    # weight class
    transformer_weight_class = None

    # infer class
    transformer_layer_infer_class = None

    # Mem manager class
    memory_manager_class = partial(MemoryManager, always_copy=True)

    def __init__(self, tp_rank, world_size, weight_dir, max_total_token_num, load_way="HF", mode=[]):
        self.init_mode(mode)
        super().__init__(tp_rank, world_size, weight_dir, max_total_token_num, load_way, mode)

    def init_mode(self, mode):
        self.q_group_size = 128
        for _mode in mode:
            if _mode.startswith('g'):
                self.q_group_size = int(_mode[1:])
        infer_class_dict = {
            'int4weight': partial(LlamaTransformerLayerInferINT4, group_size=self.q_group_size),
            'int8weight': LlamaTransformerLayerInferINT8,
        }
        for _mode in mode:
            if _mode in infer_class_dict:
                self.transformer_layer_infer_class = infer_class_dict[_mode]
                print("Model using mode", _mode)
        self.transformer_weight_class = partial(LlamaTransformerLayerWeightQuantized, group_size=self.q_group_size)
