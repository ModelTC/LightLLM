from typing_extensions import override

import torch

from lightllm.common.basemodel.layer_weights.transformer_layer_weight import TransformerLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import ROWMMWeight, NormWeight


class NSAIndexerWeight(TransformerLayerWeight):
    def __init__(self, layer_num, data_type, network_config, mode, quant_cfg):
        super().__init__(layer_num, data_type, network_config, mode, quant_cfg)
        return

    @override
    def _init_weight(self):
        prefix = f"model.layers.{self.layer_num_}.self_attn.indexer"
        
        self.wq_b_proj_ = ROWMMWeight(
            weight_name=f"{prefix}.wq_b.weight",
            data_type=self.data_type_,
            quant_cfg=self.quant_cfg,
            layer_num=self.layer_num_,
            name="wq_b",
            tp_rank=0,
            tp_world_size=1,
        )
        self.wk_proj_ = ROWMMWeight(
            weight_name=f"{prefix}.wk.weight",
            data_type=self.data_type_,
            quant_cfg=self.quant_cfg,
            layer_num=self.layer_num_,
            name="wk",
            tp_rank=0,
            tp_world_size=1,
        )
        self.k_norm_ = NormWeight(
            f"{prefix}.k_norm.weight", 
            torch.float32, 
            bias_name=f"{prefix}.k_norm.bias"
        )
        self.weights_proj_ = ROWMMWeight(
            weight_name=f"{prefix}.weights_proj.weight",
            data_type=self.data_type_,
            quant_cfg=None, 
            layer_num=self.layer_num_,
            name="weights_proj",
            tp_rank=0,
            tp_world_size=1,
        )
