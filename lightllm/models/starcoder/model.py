import os
import json
import torch

from lightllm.models.starcoder.layer_infer.transformer_layer_infer import StarcoderTransformerLayerInfer
from lightllm.models.starcoder.layer_infer.pre_layer_infer import StarcoderPreLayerInfer
from lightllm.models.starcoder.layer_infer.infer_struct import StarcoderInferStateInfo
from lightllm.models.starcoder.layer_weights.transformer_layer_weight import StarcoderTransformerLayerWeight
from lightllm.models.starcoder.layer_weights.pre_and_post_layer_weight import StarcoderPreAndPostLayerWeight

from lightllm.common.basemodel.layer_weights.hf_load_utils import load_hf_weights
from lightllm.common.mem_manager import MemoryManager
from lightllm.models.bloom.model import BloomTpPartModel
from lightllm.common.build_utils import repair_config


class StarcoderTpPartModel(BloomTpPartModel):
    # weight class
    pre_and_post_weight_class = StarcoderPreAndPostLayerWeight
    transformer_weight_class = StarcoderTransformerLayerWeight

    # infer class
    pre_layer_infer_class = StarcoderPreLayerInfer
    transformer_layer_infer_class = StarcoderTransformerLayerInfer
    infer_state_class = StarcoderInferStateInfo

    def __init__(self, tp_rank, world_size, weight_dir, max_total_token_num,
                 load_way="HF", mode=[], weight_dict=None, finetune_config=None):
        super().__init__(
            tp_rank,
            world_size,
            weight_dir,
            max_total_token_num,
            load_way,
            mode,
            weight_dict,
            finetune_config)

    def _init_config(self):
        super()._init_config()
        # rename key
        # repair_config()
        repair_config(
            self.config,
            same_names=[
                "rms_norm_eps",
                "layer_norm_epsilon"])
        self.config['num_key_value_heads'] = 1
        return

    def _verify_params(self):
        assert self.load_way == "HF", "StarCoder only support HF format to load Now!"

    def _init_mem_manager(self):
        self.mem_manager = MemoryManager(self.max_total_token_num,
                                         dtype=torch.float16,
                                         head_num=self.config["num_key_value_heads"],
                                         head_dim=self.config["hidden_size"] // self.config["num_attention_heads"],
                                         layer_num=self.config["num_hidden_layers"])
        return

    def _init_some_value(self):
        super()._init_some_value()
        self.tp_k_head_num_ = self.config["num_key_value_heads"]
        self.tp_v_head_num_ = self.config["num_key_value_heads"]

    def _init_weights(self):
        self.pre_post_weight = self.pre_and_post_weight_class(
            self.tp_rank_,
            self.world_size_,
            torch.float16,
            network_config=self.config,
            mode=self.mode)
        self.trans_layers_weight = [
            self.transformer_weight_class(
                i,
                self.tp_rank_,
                self.world_size_,
                torch.float16,
                network_config=self.config,
                mode=self.mode)
            for i in range(self.config["n_layer"])
        ]
        load_hf_weights(
            "fp16",
            weight_dir=self.weight_dir_,
            pre_post_layer=self.pre_post_weight,
            transformer_layer_list=self.trans_layers_weight,
            weight_dict=self.weight_dict)
        self.pre_post_weight.verify_load()
        [weight.verify_load() for weight in self.trans_layers_weight]
        return
