from typing_extensions import override

import torch

from lightllm.common.basemodel.layer_weights.transformer_layer_weight import TransformerLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import ROWMMWeight, LayerNormWeight


class NSAIndexerWeight(TransformerLayerWeight):
    def __init__(self, layer_num, data_type, network_config, quant_cfg):
        super().__init__(layer_num, data_type, network_config, quant_cfg)
        return

    def _parse_config(self):
        self.q_lora_rank = self.network_config_["q_lora_rank"]
        self.index_n_heads = self.network_config_["index_n_heads"]
        self.index_head_dim = self.network_config_["index_head_dim"]
        self.hidden_size = self.network_config_["hidden_size"]

    def _init_weight(self):
        prefix = f"model.layers.{self.layer_num_}.self_attn.indexer"

        self.wq_b_proj_ = ROWMMWeight(
            in_dim=self.q_lora_rank,
            out_dims=[self.index_n_heads * self.index_head_dim],
            weight_names=f"{prefix}.wq_b.weight",
            data_type=self.data_type_,
            quant_method=None,
            tp_rank=0,
            tp_world_size=1,
        )
        self.wk_proj_ = ROWMMWeight(
            in_dim=self.hidden_size,
            out_dims=[self.index_head_dim],
            weight_names=f"{prefix}.wk.weight",
            data_type=self.data_type_,
            quant_method=None,
            tp_rank=0,
            tp_world_size=1,
        )
        self.k_norm_ = LayerNormWeight(
            dim=self.index_head_dim,
            weight_name=f"{prefix}.k_norm.weight",
            data_type=self.data_type_,
            bias_name=f"{prefix}.k_norm.bias",
        )
        self.weights_proj_ = ROWMMWeight(
            in_dim=self.hidden_size,
            out_dims=[self.index_n_heads],
            weight_names=f"{prefix}.weights_proj.weight",
            data_type=self.data_type_,
            quant_method=None,
            tp_rank=0,
            tp_world_size=1,
        )
