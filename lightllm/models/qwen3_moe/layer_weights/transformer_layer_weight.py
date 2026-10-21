from lightllm.models.qwen3.layer_weights.transformer_layer_weight import Qwen3TransformerLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import ROWMMWeight, FusedMoeWeight, QKVROWNMMWeight


class Qwen3MOETransformerLayerWeight(Qwen3TransformerLayerWeight):
    def __init__(self, layer_num, data_type, network_config, quant_cfg=None):
        self.n_routed_experts = network_config.get("num_experts", 0)
        self.is_moe = (
            network_config.get("num_experts", 0) > 0
            and layer_num not in network_config.get("mlp_only_layers", [])
            and (layer_num + 1) % network_config.get("decoder_sparse_step", 1) == 0
        )
        super().__init__(layer_num, data_type, network_config, quant_cfg)
        return

    def load_hf_weights(self, weights):
        if self.is_moe:
            split_fused_expert_weights(
                weights,
                self.layer_num_,
                self.network_config_["moe_intermediate_size"],
            )
        return super().load_hf_weights(weights)

    def _init_weight_names(self):
        self._q_weight_name = f"model.layers.{self.layer_num_}.self_attn.q_proj.weight"
        self._q_norm_name = f"model.layers.{self.layer_num_}.self_attn.q_norm.weight"
        self._q_bias_name = None
        self._k_weight_name = f"model.layers.{self.layer_num_}.self_attn.k_proj.weight"
        self._k_norm_name = f"model.layers.{self.layer_num_}.self_attn.k_norm.weight"
        self._k_bias_name = None
        self._v_weight_name = f"model.layers.{self.layer_num_}.self_attn.v_proj.weight"
        self._v_bias_name = None
        self._kv_weight_name = f"model.layers.{self.layer_num_}.self_attn.kv_proj.weight"
        self._kv_bias_name = None
        self._o_weight_name = f"model.layers.{self.layer_num_}.self_attn.o_proj.weight"
        self._o_bias_name = None
        self._att_norm_weight_name = f"model.layers.{self.layer_num_}.input_layernorm.weight"
        self._att_norm_bias_name = None
        self._ffn_norm_weight_name = f"model.layers.{self.layer_num_}.post_attention_layernorm.weight"
        self._ffn_norm_bias_name = None

    def _init_weight(self):
        self._init_qkv()
        self._init_o()
        if self.is_moe:
            self._init_moe()
        else:
            self._init_ffn()
        self._init_norm()

    def _init_moe(self):
        moe_intermediate_size = self.network_config_["moe_intermediate_size"]
        self.moe_gate = ROWMMWeight(
            in_dim=self.network_config_["hidden_size"],
            out_dims=[self.n_routed_experts],
            weight_names=f"model.layers.{self.layer_num_}.mlp.gate.weight",
            data_type=self.data_type_,
            quant_method=None,
            tp_rank=0,
            tp_world_size=1,
        )
        self.experts = FusedMoeWeight(
            gate_proj_name="gate_proj",
            down_proj_name="down_proj",
            up_proj_name="up_proj",
            e_score_correction_bias_name="",
            weight_prefix=f"model.layers.{self.layer_num_}.mlp.experts",
            n_routed_experts=self.n_routed_experts,
            hidden_size=self.network_config_["hidden_size"],
            moe_intermediate_size=moe_intermediate_size,
            data_type=self.data_type_,
            quant_method=self.quant_cfg.get_quant_method(self.layer_num_, "fused_moe"),
            layer_num=self.layer_num_,
            network_config=self.network_config_,
        )

    def _init_qkv(self):
        in_dim = self.n_embed
        self.qkv_proj = QKVROWNMMWeight(
            in_dim=in_dim,
            q_head_num=self.q_head_num_,
            kv_head_num=self.k_head_num_,
            head_dim=self.head_dim,
            weight_names=[self._q_weight_name, self._k_weight_name, self._v_weight_name],
            data_type=self.data_type_,
            bias_names=[self._q_bias_name, self._k_bias_name, self._v_bias_name],
            quant_method=self.get_quant_method("qkv_proj"),
        )


def split_fused_expert_weights(weights: dict, layer_num: int, moe_intermediate_size: int):
    """将 HF 打包的 fused MoE expert 权重拆成按 expert 索引的独立权重。

    部分 checkpoint（如 Qwen3-MoE）把所有 expert 的 gate/up/down 压成
    ``mlp.experts.{gate_up,gate,up,down}_proj`` 的打包张量
    （首维为 expert 数）。本函数只处理 ``model.layers.{layer_num}`` 下的这类
    key：弹出打包权重，再写入
    ``mlp.experts.{expert_idx}.{gate,up,down}_proj.weight``，供后续按 expert
    加载。若存在 fused ``gate_up_proj``，还会按 ``moe_intermediate_size``
    沿 intermediate 维切成 gate / up。
    """
    layer_prefix = f"model.layers.{layer_num}."
    keys = list(weights.keys())

    for k in keys:
        if not k.startswith(layer_prefix):
            continue

        if "mlp.experts.gate_up_proj" in k:
            fused_weight = weights.pop(k)
            prefix = k.rsplit(".gate_up_proj", 1)[0]
            gate_weight = fused_weight[:, :moe_intermediate_size, :]
            up_weight = fused_weight[:, moe_intermediate_size:, :]

            for expert_idx in range(fused_weight.shape[0]):
                weights[f"{prefix}.{expert_idx}.gate_proj.weight"] = gate_weight[expert_idx]
                weights[f"{prefix}.{expert_idx}.up_proj.weight"] = up_weight[expert_idx]

        elif "mlp.experts.gate_proj" in k:
            gate_weight = weights.pop(k)
            prefix = k.rsplit(".gate_proj", 1)[0]
            for expert_idx in range(gate_weight.shape[0]):
                weights[f"{prefix}.{expert_idx}.gate_proj.weight"] = gate_weight[expert_idx]

        elif "mlp.experts.up_proj" in k:
            up_weight = weights.pop(k)
            prefix = k.rsplit(".up_proj", 1)[0]
            for expert_idx in range(up_weight.shape[0]):
                weights[f"{prefix}.{expert_idx}.up_proj.weight"] = up_weight[expert_idx]

        elif "mlp.experts.down_proj" in k:
            down_weight = weights.pop(k)
            prefix = k.rsplit(".down_proj", 1)[0]
            for expert_idx in range(down_weight.shape[0]):
                weights[f"{prefix}.{expert_idx}.down_proj.weight"] = down_weight[expert_idx]
