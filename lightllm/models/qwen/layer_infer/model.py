import os
import json
import torch
from lightllm.models.llama.layer_infer.pre_layer_inference import PreLayerInfer
from lightllm.models.llama.layer_infer.post_layer_inference import PostLayerInfer
from lightllm.models.qwen.layer_infer.transformer_layer_inference import QwenTransformerLayerInfer
from lightllm.models.qwen.layer_weights.pre_and_post_layer_weight import *
from lightllm.models.qwen.layer_weights.transformer_layer_weight import *
from lightllm.models.llama.layer_weights.hf_load_utils import load_hf_weights
from lightllm.models.llama.layer_infer.model import LlamaTpPartModel
from lightllm.common.mem_manager import MemoryManager
from lightllm.common.int8kv_mem_manager import INT8KVMemoryManager
from lightllm.common.infer_utils import init_bloc

class QWenTpPartModel(LlamaTpPartModel):
    def __init__(self, tp_rank, world_size, weight_dir, max_total_token_num, load_way="HF", mode=""):
        self.tp_rank_ = tp_rank
        self.world_size_ = world_size
        self.weight_dir_ = weight_dir
        with open(os.path.join(weight_dir, "config.json"), 'r') as json_file:
            self.config = json.load(json_file)

        self.config["num_attention_heads"] = self.config["n_head"]
        self.config["hidden_size"] = self.config["n_embd"]
        self.config["num_hidden_layers"] = self.config["n_layer"]
        self.config["rms_norm_eps"] = self.config["layer_norm_epsilon"]

        assert load_way == "HF", "llama only support HF format to load Now!"
        assert mode in ["", "int8kv"], "now support int8kv, future to support int8 int4 ..."

        mem_dict = {
            "" : MemoryManager,
            "int8kv" : INT8KVMemoryManager
        }
        
        self.mem_manager = mem_dict[mode](max_total_token_num, 
                                         dtype=torch.float16,
                                         head_num=self.config["num_attention_heads"] // world_size,
                                         head_dim=self.config["hidden_size"] // self.config["num_attention_heads"],
                                         layer_num=self.config["num_hidden_layers"])

        self.pre_post_weight = PreAndPostLayerWeight(self.tp_rank_, self.world_size_, torch.float16, self.config)
        self.trans_layers_weight = [
            TransformerLayerWeight(i, self.tp_rank_, self.world_size_, torch.float16, self.config, mode=mode)
            for i in range(self.config["num_hidden_layers"])
        ]

        load_hf_weights("fp16", weight_dir, pre_post_layer=self.pre_post_weight, transformer_layer_list=self.trans_layers_weight)

        self.pre_infer = PreLayerInfer(tp_rank=self.tp_rank_, world_size=self.world_size_, network_config=self.config)
        self.post_infer = PostLayerInfer(tp_rank=self.tp_rank_, world_size=self.world_size_, network_config=self.config)
        self.layers_infer = [
            QwenTransformerLayerInfer(
                i,
                tp_rank=self.tp_rank_,
                world_size=self.world_size_,
                network_config=self.config,
                mode=mode) for i in range(
                self.config["num_hidden_layers"])]

        self.head_num_ = self.config["num_attention_heads"]
        self.head_dim_ = self.config["hidden_size"] // self.head_num_
        assert self.head_num_ % self.world_size_ == 0
        self.tp_head_num_ = self.head_num_ // self.world_size_
        self.vocab_size = self.config["vocab_size"]
        self.init_to_get_rotary()
        return
