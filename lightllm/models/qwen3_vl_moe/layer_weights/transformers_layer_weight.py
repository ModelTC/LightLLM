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

    def load_hf_weights(self, weights):
        moe_prefix = f"model.language_model.layers.{self.layer_num_}.mlp.experts"
        gate_up_name = f"{moe_prefix}.gate_up_proj"
        down_name = f"{moe_prefix}.down_proj"

        if gate_up_name in weights:
            gate_up = weights[gate_up_name]  # [E, H, 2I]
            E, H, twoI = gate_up.shape
            assert twoI % 2 == 0, f"gate_up_proj last dim must be even, but got {twoI}"
            I_dim = twoI // 2

            if down_name in weights:
                down = weights[down_name]  # [E, I, H]
            else:
                down = None

            for e in range(E):
                gate_up_e = gate_up[e]
                gate_e = gate_up_e[:, :I_dim].transpose(0, 1).contiguous()
                up_e = gate_up_e[:, I_dim:].transpose(0, 1).contiguous()

                gate_key = f"{moe_prefix}.{e}.gate_proj.weight"
                up_key = f"{moe_prefix}.{e}.up_proj.weight"
                weights[gate_key] = gate_e
                weights[up_key] = up_e

                if down is not None:
                    down_key = f"{moe_prefix}.{e}.down_proj.weight"
                    weights[down_key] = down[e].transpose(0, 1).contiguous()

            del weights[gate_up_name]
            if down_name in weights:
                del weights[down_name]
        super().load_hf_weights(weights)

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
            )
        else:
            raise ValueError(f"Unsupported moe mode: {moe_mode}")
