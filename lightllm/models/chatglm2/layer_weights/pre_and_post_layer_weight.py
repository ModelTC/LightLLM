import torch
import numpy as np
from lightllm.models.llama.layer_weights.pre_and_post_layer_weight import LlamaPreAndPostLayerWeight


class ChatGLM2PreAndPostLayerWeight(LlamaPreAndPostLayerWeight):
    def __init__(self, tp_rank, world_size, data_type, network_config, mode):
        super().__init__(tp_rank, world_size, data_type, network_config, mode)

    def load_hf_weights(self, weights):
        # input layernorm params

        vob_size = self.network_config_["padded_vocab_size"]
        split_vob_size = vob_size // self.world_size_
        n_embed = self.network_config_["hidden_size"]
        if "transformer.embedding.word_embeddings.weight" in weights:
            self.wte_weight_ = weights['transformer.embedding.word_embeddings.weight'][split_vob_size *
                                                                    self.tp_rank_: split_vob_size * (self.tp_rank_ + 1), :].contiguous().to(self.data_type_).cuda()
        if 'transformer.output_layer.weight' in weights:
            self.lm_head_weight_ = weights['transformer.output_layer.weight'][split_vob_size * self.tp_rank_: split_vob_size *
                                                            (self.tp_rank_ + 1), :].contiguous().to(self.data_type_).cuda()
        if "transformer.encoder.final_layernorm.weight" in weights:
            self.final_norm_weight_ = weights['transformer.encoder.final_layernorm.weight'].contiguous().to(self.data_type_).cuda()

        return
