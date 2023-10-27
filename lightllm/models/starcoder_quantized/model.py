import os
import json
import torch
from functools import partial

from lightllm.models.starcoder.layer_infer.transformer_layer_infer import StarcoderTransformerLayerInfer
from lightllm.models.starcoder.layer_infer.pre_layer_infer import StarcoderPreLayerInfer
from lightllm.models.starcoder.layer_infer.infer_struct import StarcoderInferStateInfo
from lightllm.models.starcoder.layer_weights.transformer_layer_weight import StarcoderTransformerLayerWeight
from lightllm.models.starcoder.layer_weights.pre_and_post_layer_weight import StarcoderPreAndPostLayerWeight

from lightllm.common.basemodel.layer_weights.hf_load_utils import load_hf_weights
from lightllm.common.mem_manager import MemoryManager
from lightllm.models.bloom.model import BloomTpPartModel
from lightllm.common.build_utils import repair_config
from lightllm.models.starcoder.model import StarcoderTpPartModel
from lightllm.models.starcoder_quantized.layer_infer.transformer_layer_infer import StarcoderTransformerLayerInferINT8, StarcoderTransformerLayerInferINT4
from lightllm.models.starcoder_quantized.layer_weights.transformer_layer_weight import \
    StarcoderTransformerLayerWeightQuantized


class StarcoderTpPartModelQuantized(StarcoderTpPartModel):
    # weight class
    transformer_weight_class = None

    # infer class
    transformer_layer_infer_class = None

    # Mem manager class
    memory_manager_class = partial(MemoryManager, always_copy=True)

    def __init__(self, tp_rank, world_size, weight_dir, max_total_token_num, load_way="HF", mode=[], weight_dict=None,
                 finetune_config=None):
        self.init_mode(mode)
        super().__init__(tp_rank, world_size, weight_dir, max_total_token_num, load_way, mode, weight_dict,
                         finetune_config)

    def init_mode(self, mode):
        self.q_group_size = 128
        for _mode in mode:
            if _mode.startswith('g'):
                self.q_group_size = int(_mode[1:])
        infer_class_dict = {
            'int4weight': partial(StarcoderTransformerLayerInferINT4, group_size=self.q_group_size),
            'int8weight': StarcoderTransformerLayerInferINT8,
        }
        for _mode in mode:
            if _mode in infer_class_dict:
                self.transformer_layer_infer_class = infer_class_dict[_mode]
                print("Model using mode", _mode)
        self.transformer_weight_class = partial(StarcoderTransformerLayerWeightQuantized, group_size=self.q_group_size)
