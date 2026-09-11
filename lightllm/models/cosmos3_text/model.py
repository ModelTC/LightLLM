from __future__ import annotations

import json
import os

from lightllm.common.basemodel.layer_weights.hf_load_utils import load_hf_weights
from lightllm.common.build_utils import repair_config
from lightllm.models.cosmos3_text.layer_weights import (
    Cosmos3TextPreAndPostLayerWeight,
    Cosmos3TextTransformerLayerWeight,
)
from lightllm.models.qwen3.model import Qwen3TpPartModel
from lightllm.models.registry import ModelRegistry


@ModelRegistry("cosmos3_omni")
class Cosmos3TextTpPartModel(Qwen3TpPartModel):
    """Cosmos3 causal text tower using the standard LightLLM runtime."""

    pre_and_post_weight_class = Cosmos3TextPreAndPostLayerWeight
    transformer_weight_class = Cosmos3TextTransformerLayerWeight

    def _init_config(self):
        with open(os.path.join(self.weight_dir_, "config.json"), "r") as config_file:
            self.config = json.load(config_file)["text_config"]
        repair_config(self.config, same_names=["num_attention_heads", "n_head"])
        repair_config(self.config, same_names=["hidden_size", "n_embd", "n_embed"])
        repair_config(self.config, same_names=["num_hidden_layers", "n_layer"])
        if self.finetune_config:
            self.config["vocab_size"] = self.finetune_config.vocab_size

    def _load_hf_weights(self):
        # The official root index also references the VAE and vision encoder.
        # Scanning transformer/ confines loading to the joint-transformer shards;
        # the weight objects above select only text keys from those shards.
        transformer_weight_dir = os.path.join(self.weight_dir_, "transformer")
        if self.weight_dict is None and not os.path.isdir(transformer_weight_dir):
            raise FileNotFoundError(
                f"Cosmos3 text target expects official transformer weights under {transformer_weight_dir}"
            )
        load_hf_weights(
            self.data_type,
            weight_dir=transformer_weight_dir,
            pre_post_layer=self.pre_post_weight,
            transformer_layer_list=self.trans_layers_weight,
            weight_dict=self.weight_dict,
        )
        self.pre_post_weight.verify_load()
        for layer_weight in self.trans_layers_weight:
            layer_weight.verify_load()
