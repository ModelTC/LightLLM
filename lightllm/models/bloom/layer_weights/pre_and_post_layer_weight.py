import torch
import numpy as np
from .base_layer_weight import BaseLayerWeight


class PreAndPostLayerWeight(BaseLayerWeight):
    def __init__(self, tp_rank, world_size, data_type, network_config):
        self.tp_rank_ = tp_rank
        self.data_type_ = data_type
        self.world_size_ = world_size
        self.network_config_ = network_config

    def load_ft_weights(self, weight_dir=None):
        # input layernorm params
        self.pre_layernorm_weight_ = self.load_to_torch(f"{weight_dir}/model.pre_decoder_layernorm.weight.bin").cuda()
        self.pre_layernorm_bias_ = self.load_to_torch(f"{weight_dir}/model.pre_decoder_layernorm.bias.bin").cuda()

        self.final_layernorm_weight_ = self.load_to_torch(f"{weight_dir}/model.final_layernorm.weight.bin").cuda()
        self.final_layernorm_bias_ = self.load_to_torch(f"{weight_dir}/model.final_layernorm.bias.bin").cuda()

        vob_size = self.network_config_["vocab_size"]
        split_vob_size = vob_size // self.world_size_
        n_embed = self.network_config_["n_embed"]
        wte_weight = self.load_to_torch(f"{weight_dir}/model.wte.bin").reshape(vob_size, n_embed)
        wte_weight = wte_weight[split_vob_size * self.tp_rank_: split_vob_size * (self.tp_rank_ + 1), :].contiguous().cuda()
        self.wte_weight_ = wte_weight
        return

    def load_hf_weights(self, weights):
        if isinstance(self.data_type_, str):
            if self.data_type_ == "fp16":
                self.data_type_ = torch.float16
            elif self.data_type_ == "fp32":
                self.data_type_ = torch.float32
            else:
                raise
        if "word_embeddings_layernorm.weight" in weights:
            self.pre_layernorm_weight_ = weights['word_embeddings_layernorm.weight'].contiguous().to(self.data_type_).cuda()
        if "word_embeddings_layernorm.bias" in weights:
            self.pre_layernorm_bias_ = weights['word_embeddings_layernorm.bias'].contiguous().to(self.data_type_).cuda()
        if "ln_f.weight" in weights:
            self.final_layernorm_weight_ = weights['ln_f.weight'].contiguous().to(self.data_type_).cuda()
        if "ln_f.bias" in weights:
            self.final_layernorm_bias_ = weights["ln_f.bias"].contiguous().to(self.data_type_).cuda()
        if "word_embeddings.weight" in weights:
            vob_size = self.network_config_["vocab_size"]
            split_vob_size = vob_size // self.world_size_
            self.wte_weight_ = weights["word_embeddings.weight"][split_vob_size *
                                                                 self.tp_rank_: split_vob_size * (self.tp_rank_ + 1), :].contiguous().cuda()
        return
