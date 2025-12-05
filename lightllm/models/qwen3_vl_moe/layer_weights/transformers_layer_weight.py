import os
from lightllm.models.qwen3_moe.layer_weights.transformer_layer_weight import Qwen3MOETransformerLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import ROWMMWeight, FusedMoeWeightEP, create_tp_moe_wegiht_obj


class Qwen3VLMOETransformerLayerWeight(Qwen3MOETransformerLayerWeight):
    def __init__(self, layer_num, data_type, network_config, mode=[], quant_cfg=None):
        super().__init__(layer_num, data_type, network_config, mode, quant_cfg)

    def _init_weight_names(self):
        super()._init_weight_names()
        self._q_weight_name = f"model.language_model.layers.{self.layer_num_}.self_attn.q_proj.weight"
        self._q_norm_name = f"model.language_model.layers.{self.layer_num_}.self_attn.q_norm.weight"
        self._q_bias_name = None
        self._k_weight_name = f"model.language_model.layers.{self.layer_num_}.self_attn.k_proj.weight"
        self._k_norm_name = f"model.language_model.layers.{self.layer_num_}.self_attn.k_norm.weight"
        self._k_bias_name = None
        self._v_weight_name = f"model.language_model.layers.{self.layer_num_}.self_attn.v_proj.weight"
        self._v_bias_name = None
        self._o_weight_name = f"model.language_model.layers.{self.layer_num_}.self_attn.o_proj.weight"
        self._o_bias_name = None
        self._att_norm_weight_name = f"model.language_model.layers.{self.layer_num_}.input_layernorm.weight"
        self._att_norm_bias_name = None
        self._ffn_norm_weight_name = f"model.language_model.layers.{self.layer_num_}.post_attention_layernorm.weight"
        self._ffn_norm_bias_name = None

    def _init_moe(self):
        moe_intermediate_size = self.network_config_["moe_intermediate_size"]
        self.moe_gate = ROWMMWeight(
            weight_names=f"model.language_model.layers.{self.layer_num_}.mlp.gate.weight",
            data_type=self.data_type_,
            layer_num=self.layer_num_,
            name="moe_gate",
            tp_rank=0,
            tp_world_size=1,
        )
        moe_mode = os.getenv("MOE_MODE", "TP")
        assert moe_mode in ["EP", "TP"]

        if moe_mode == "TP":
            self.experts = create_tp_moe_wegiht_obj(
                gate_proj_name="gate_proj",
                down_proj_name="down_proj",
                up_proj_name="up_proj",
                e_score_correction_bias_name="",
                weight_prefix=f"model.language_model.layers.{self.layer_num_}.mlp.experts",
                n_routed_experts=self.n_routed_experts,
                split_inter_size=moe_intermediate_size // self.tp_world_size_,
                data_type=self.data_type_,
                network_config=self.network_config_,
                layer_num=self.layer_num_,
                quant_cfg=self.quant_cfg,
                num_fused_shared_experts=0,
                fused_gate_up=True,
                gate_up_proj_name="gate_up_proj",
            )
        elif moe_mode == "EP":
            self.experts = FusedMoeWeightEP(
                gate_proj_name="gate_proj",
                down_proj_name="down_proj",
                up_proj_name="up_proj",
                e_score_correction_bias_name="",
                weight_prefix=f"model.language_model.layers.{self.layer_num_}.mlp.experts",
                n_routed_experts=self.n_routed_experts,
                data_type=self.data_type_,
                network_config=self.network_config_,
                layer_num=self.layer_num_,
                quant_cfg=self.quant_cfg,
                fused_gate_up=True,
                gate_up_proj_name="gate_up_proj",
            )
        else:
            raise ValueError(f"Unsupported moe mode: {moe_mode}")
