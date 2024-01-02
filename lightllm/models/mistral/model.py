import os
import json
import torch

from lightllm.common.basemodel import TpPartBaseModel
from lightllm.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight
from lightllm.models.llama.layer_weights.pre_and_post_layer_weight import LlamaPreAndPostLayerWeight
from lightllm.models.llama.layer_infer.pre_layer_infer import LlamaPreLayerInfer
from lightllm.models.llama.layer_infer.post_layer_infer import LlamaPostLayerInfer
from lightllm.models.mistral.infer_struct import MistralInferStateInfo
from lightllm.models.mistral.layer_infer.transformer_layer_infer import MistralTransformerLayerInfer
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer

from lightllm.common.mem_utils import MemoryManager
from lightllm.common.infer_utils import init_req_to_token_indexes
from lightllm.common.basemodel.triton_kernel.copy_kv_index_to_req import copy_kv_index_to_req

class MistralTpPartModel(TpPartBaseModel):
    # weight class
    pre_and_post_weight_class = LlamaPreAndPostLayerWeight
    transformer_weight_class = LlamaTransformerLayerWeight

    # infer class
    pre_layer_infer_class = LlamaPreLayerInfer
    post_layer_infer_class = LlamaPostLayerInfer
    transformer_layer_infer_class = MistralTransformerLayerInfer

    # infer state class
    infer_state_class = MistralInferStateInfo

    def __init__(self, kvargs):
        super().__init__(kvargs)
        return

    def _init_config(self):
        super()._init_config()
        if self.config["sliding_window"] == None:
            self.config["sliding_window"] = self.max_total_token_num
        # rename key [SYM: to be confirmed]
        return

    def _verify_params(self):
        assert self.load_way in ["HF"], "mistral only supports HF format to load Now!"
        assert self.config["num_key_value_heads"] % self.world_size_ == 0
        assert self.config["num_attention_heads"] % self.world_size_ == 0
        return

    def _init_custom(self):
        self._init_to_get_rotary()
        return

    def _init_mem_manager(self):
        self.mem_manager = MemoryManager(self.max_total_token_num, # [SYM] should be sliding window?
                                        dtype=torch.float16,
                                        head_num=self.config["num_key_value_heads"] // self.world_size_,
                                        head_dim=self.config["hidden_size"] // self.config["num_attention_heads"],
                                        layer_num=self.config["num_hidden_layers"],
                                        always_copy=False)       
        return

    def _init_to_get_rotary(self, default_base=10000):
        if self.config.get("rope_scaling", {}) is None:
            rope_scaling_factor = 1.0
        else:
            rope_scaling_factor = self.config.get("rope_scaling", {}).get("factor", 1.0)

        base = self.config.get("rope_theta", float(default_base))

        if "max_sequence_length" in self.config:
            max_seq_len = self.config["max_sequence_length"]
        else:
            max_position_embeddings = self.config.get(
                "max_position_embeddings",
                2048 if base <= 10000.0 + 1e-5 else 16384
            )
            max_seq_len = max_position_embeddings * rope_scaling_factor

        inv_freq = 1.0 / (base ** (torch.arange(0, self.head_dim_, 2, device="cpu", dtype=torch.float32) / self.head_dim_))
        t = torch.arange(max_seq_len + 1024 * 64, device="cpu", dtype=torch.float32) / rope_scaling_factor
        freqs = torch.outer(t, inv_freq)

        self._cos_cached = torch.cos(freqs).to(torch.float16).cuda()
        self._sin_cached = torch.sin(freqs).to(torch.float16).cuda()
        return
    