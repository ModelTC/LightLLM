import os
import json
import torch
from functools import partial

from lightllm.common.build_utils import repair_config
from lightllm.models.starcoder.model import StarcoderTpPartModel
from lightllm.models.starcoder_wquant.layer_infer.transformer_layer_infer import StarcoderTransformerLayerInferWQuant
from lightllm.models.starcoder_wquant.layer_weights.transformer_layer_weight import \
    StarcoderTransformerLayerWeightQuantized
from lightllm.common.mem_utils import select_mem_manager_class


class StarcoderTpPartModelWQuant(StarcoderTpPartModel):
    # weight class
    transformer_weight_class = StarcoderTransformerLayerWeightQuantized

    # infer class
    transformer_layer_infer_class = StarcoderTransformerLayerInferWQuant

    def __init__(self, kvargs):
        super().__init__(kvargs)

    def _verify_params(self):
        assert self.load_way in ["HF", "DS"], "llama only supports HF and DS format to load Now!"
        assert any("w6a16" in mode_ or "w4a16" in mode_ or "w8a16" in mode_ for mode_ in self.mode), "only for weight quant model"
        return
    
    def _init_mem_manager(self):
        self.mem_manager = select_mem_manager_class(self.mode)(self.max_total_token_num, 
                                                     dtype=torch.float16,
                                                     head_num=self.config["num_key_value_heads"],
                                                     head_dim=self.config["hidden_size"] // self.config["num_attention_heads"],
                                                     layer_num=self.config["num_hidden_layers"],
                                                     always_copy=True)
        return