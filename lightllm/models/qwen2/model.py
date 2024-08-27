import os
import json
import torch

from lightllm.common.basemodel import TpPartBaseModel
from lightllm.models.llama.layer_infer.pre_layer_infer import LlamaPreLayerInfer
from lightllm.models.llama.layer_infer.post_layer_infer import LlamaPostLayerInfer
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.models.llama.splitfuse_infer_struct import LlamaSplitFuseInferStateInfo
from lightllm.models.qwen2.layer_weights.pre_and_post_layer_weight import Qwen2PreAndPostLayerWeight
from lightllm.models.qwen2.layer_weights.transformer_layer_weight import Qwen2TransformerLayerWeight
from lightllm.models.qwen2.layer_infer.transformer_layer_infer import Qwen2TransformerLayerInfer

from lightllm.common.mem_utils import MemoryManager
from lightllm.common.infer_utils import init_req_to_token_indexes
from lightllm.common.basemodel.triton_kernel.copy_kv_index_to_req import copy_kv_index_to_req
from lightllm.models.llama.model import LlamaTpPartModel


class Qwen2TpPartModel(LlamaTpPartModel):
    # weight class
    pre_and_post_weight_class = Qwen2PreAndPostLayerWeight
    transformer_weight_class = Qwen2TransformerLayerWeight

    # infer class
    pre_layer_infer_class = LlamaPreLayerInfer
    post_layer_infer_class = LlamaPostLayerInfer
    transformer_layer_infer_class = Qwen2TransformerLayerInfer

    # infer state class
    infer_state_class = LlamaInferStateInfo
    splitfuse_infer_state_class = LlamaSplitFuseInferStateInfo

    def __init__(self, kvargs):
        super().__init__(kvargs)
        return

    def _init_config(self):
        super()._init_config()
        if self.config["sliding_window"] is None:
            self.config["sliding_window"] = self.max_total_token_num
        # rename key [SYM: to be confirmed]
        return

    def _verify_params(self):
        assert self.load_way in ["HF"], "mistral only supports HF format to load Now!"
        assert self.config["num_key_value_heads"] % self.world_size_ == 0
        assert self.config["num_attention_heads"] % self.world_size_ == 0
        return

    def _init_mem_manager(self):
        self.mem_manager = MemoryManager(
            self.max_total_token_num,  # [SYM] should be sliding window?
            dtype=self.data_type,
            head_num=self.config["num_key_value_heads"] // self.world_size_,
            head_dim=self.config["hidden_size"] // self.config["num_attention_heads"],
            layer_num=self.config["num_hidden_layers"],
            always_copy=False,
        )
        return
