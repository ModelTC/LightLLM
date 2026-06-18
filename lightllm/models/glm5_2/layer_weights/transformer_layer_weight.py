import torch

from lightllm.common.basemodel.layer_weights.meta_weights import FusedMoeWeight, ROWMMWeight
from lightllm.models.deepseek2.layer_weights.transformer_layer_weight import Deepseek2TransformerLayerWeight
from lightllm.models.deepseek3_2.layer_weights.transformer_layer_weight import Deepseek3_2TransformerLayerWeight
from lightllm.models.glm5_2.indexshare import owns_indexer_layer


class Glm5_2TransformerLayerWeight(Deepseek3_2TransformerLayerWeight):
    def _parse_config(self):
        super()._parse_config()
        self.has_indexer = owns_indexer_layer(self.layer_num_, self.network_config_)

    def _init_weight(self):
        Deepseek2TransformerLayerWeight._init_weight(self)
        if self.has_indexer:
            self._init_indexer_weight()

    def _init_moe(self):
        if self.num_fused_shared_experts == 0:
            self._load_mlp(f"model.layers.{self.layer_num_}.mlp.shared_experts", is_shared_experts=True)

        self.moe_gate = ROWMMWeight(
            in_dim=self.n_embed,
            out_dims=[self.n_routed_experts],
            weight_names=f"model.layers.{self.layer_num_}.mlp.gate.weight",
            data_type=torch.float32,
            quant_method=None,
            tp_rank=0,
            tp_world_size=1,
        )
        self.experts = FusedMoeWeight(
            gate_proj_name="gate_proj",
            down_proj_name="down_proj",
            up_proj_name="up_proj",
            e_score_correction_bias_name=self.e_score_correction_bias_name,
            weight_prefix=f"model.layers.{self.layer_num_}.mlp.experts",
            n_routed_experts=self.n_routed_experts,
            hidden_size=self.n_embed,
            moe_intermediate_size=self.network_config_["moe_intermediate_size"],
            data_type=self.data_type_,
            quant_method=self.quant_cfg.get_quant_method(self.layer_num_, "fused_moe"),
            num_fused_shared_experts=self.num_fused_shared_experts,
            layer_num=self.layer_num_,
            network_config=self.network_config_,
        )
