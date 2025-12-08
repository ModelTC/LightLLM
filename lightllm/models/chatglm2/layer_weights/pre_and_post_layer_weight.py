import torch
import numpy as np
from lightllm.models.llama.layer_weights.pre_and_post_layer_weight import LlamaPreAndPostLayerWeight


class ChatGLM2PreAndPostLayerWeight(LlamaPreAndPostLayerWeight):
    def __init__(self, data_type, network_config, mode):
        super().__init__(data_type, network_config, mode)

    def load_hf_weights(self, weights):
        # input layernorm params

        vob_size = self.network_config_["padded_vocab_size"]
        split_vob_size = vob_size // self.tp_world_size_
        if "transformer.embedding.word_embeddings.weight" in weights:
            wte_weight = weights["transformer.embedding.word_embeddings.weight"]
            self.wte_weight_.copy_(wte_weight[
                split_vob_size * self.tp_rank_ : split_vob_size * (self.tp_rank_ + 1), :
            ])
        if "transformer.output_layer.weight" in weights:
            lm_head_weight = weights["transformer.output_layer.weight"]
            self.lm_head_weight_.copy_(lm_head_weight[
                split_vob_size * self.tp_rank_ : split_vob_size * (self.tp_rank_ + 1), :
            ])
        if "transformer.encoder.final_layernorm.weight" in weights:
            self.final_norm_weight_.copy_(weights["transformer.encoder.final_layernorm.weight"])

        return
