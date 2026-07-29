import torch

from lightllm.common.basemodel.layer_weights.meta_weights import (
    COLMMWeight,
    FusedMoeWeight,
    GatedRMSNormWeight,
    ParameterWeight,
    RMSNormWeight,
    ROWMMWeight,
    TpParameterWeight,
)
from lightllm.models.deepseek2.layer_weights.transformer_layer_weight import (
    Deepseek2TransformerLayerWeight,
)
from lightllm.models.kimi_linear.attnres import (
    BlockAttnResConfig,
    normalize_attnres_query_weight,
)
from lightllm.utils.envs_utils import get_env_start_args


class KimiLinearTransformerLayerWeight(Deepseek2TransformerLayerWeight):
    def __init__(self, layer_num, data_type, network_config, quant_cfg=None):
        self.is_linear_attention_layer = (layer_num + 1) in network_config["linear_attn_config"]["kda_layers"]
        super().__init__(layer_num, data_type, network_config, quant_cfg)

    def _parse_config(self):
        super()._parse_config()
        self.use_gated_mla = bool(self.network_config_.get("mla_use_output_gate", False))
        self.moe_latent_size = self.network_config_.get("routed_expert_hidden_size")
        if self.moe_latent_size is not None:
            if (
                not isinstance(self.moe_latent_size, int)
                or isinstance(self.moe_latent_size, bool)
                or self.moe_latent_size <= 0
            ):
                raise ValueError("routed_expert_hidden_size must be a positive integer")
            self.num_fused_shared_experts = 0
        self.latent_moe_use_norm = bool(self.network_config_.get("latent_moe_use_norm", False))
        self.hidden_act = self.network_config_.get("hidden_act", "silu")
        if self.hidden_act not in ("silu", "situ"):
            raise ValueError(f"unsupported Kimi activation: {self.hidden_act}")
        self.activation_situ_beta = self.network_config_.get("activation_situ_beta", 1.0)
        self.activation_situ_linear_beta = self.network_config_.get("activation_situ_linear_beta")
        self.attnres_config = BlockAttnResConfig.from_network_config(self.network_config_)

    def _init_weight(self):
        if self.is_linear_attention_layer:
            self._init_kda()
        else:
            self._init_qkvo()
            if self.use_gated_mla:
                self._init_mla_output_gate()
        if self.is_moe:
            self._init_moe()
        else:
            self._init_ffn()
        self._init_norm()
        if self.attnres_config is not None:
            self._init_attnres()

    def _init_attnres(self):
        layer_num = self.layer_num_
        hidden_size = self.n_embed
        self.attnres_attn_query = ParameterWeight(
            weight_name=f"model.layers.{layer_num}.self_attention_res_proj.weight",
            data_type=self.data_type_,
            weight_shape=(hidden_size,),
        )
        self.attnres_attn_norm = RMSNormWeight(
            dim=hidden_size,
            weight_name=f"model.layers.{layer_num}.self_attention_res_norm.weight",
            data_type=self.data_type_,
        )
        self.attnres_mlp_query = ParameterWeight(
            weight_name=f"model.layers.{layer_num}.mlp_res_proj.weight",
            data_type=self.data_type_,
            weight_shape=(hidden_size,),
        )
        self.attnres_mlp_norm = RMSNormWeight(
            dim=hidden_size,
            weight_name=f"model.layers.{layer_num}.mlp_res_norm.weight",
            data_type=self.data_type_,
        )
        if layer_num + 1 == self.network_config_["num_hidden_layers"]:
            self.attnres_final_query = ParameterWeight(
                weight_name="model.output_attn_res_proj.weight",
                data_type=self.data_type_,
                weight_shape=(hidden_size,),
            )
            self.attnres_final_norm = RMSNormWeight(
                dim=hidden_size,
                weight_name="model.output_attn_res_norm.weight",
                data_type=self.data_type_,
            )

    def _init_kda(self):
        prefix = f"model.layers.{self.layer_num_}.self_attn"
        config = self.network_config_["linear_attn_config"]
        hidden_size = self.network_config_["hidden_size"]
        head_dim = config["head_dim"]
        num_heads = config["num_heads"]
        projection_size = head_dim * num_heads
        conv_size = config["short_conv_kernel_size"]
        self.use_full_rank_gate = bool(config.get("use_full_rank_gate", False))

        self.kda_qkv_proj = ROWMMWeight(
            in_dim=hidden_size,
            out_dims=[projection_size, projection_size, projection_size],
            weight_names=[
                f"{prefix}.q_proj.weight",
                f"{prefix}.k_proj.weight",
                f"{prefix}.v_proj.weight",
            ],
            data_type=self.data_type_,
            quant_method=self.get_quant_method("qkv_proj"),
        )
        self.kda_f_a_proj = ROWMMWeight(
            in_dim=hidden_size,
            out_dims=[head_dim],
            weight_names=f"{prefix}.f_a_proj.weight",
            data_type=self.data_type_,
            quant_method=self.get_quant_method("f_a_proj"),
            tp_rank=0,
            tp_world_size=1,
        )
        self.kda_f_b_proj = ROWMMWeight(
            in_dim=head_dim,
            out_dims=[projection_size],
            weight_names=f"{prefix}.f_b_proj.weight",
            data_type=self.data_type_,
            quant_method=self.get_quant_method("f_b_proj"),
        )
        self.kda_b_proj = ROWMMWeight(
            in_dim=hidden_size,
            out_dims=[num_heads],
            weight_names=f"{prefix}.b_proj.weight",
            data_type=self.data_type_,
            quant_method=self.get_quant_method("b_proj"),
        )
        self.kda_conv1d = ROWMMWeight(
            in_dim=conv_size,
            out_dims=[projection_size, projection_size, projection_size],
            weight_names=[
                f"{prefix}.q_conv1d.weight",
                f"{prefix}.k_conv1d.weight",
                f"{prefix}.v_conv1d.weight",
            ],
            data_type=self.data_type_,
            quant_method=None,
        )
        self.kda_A_log = TpParameterWeight(
            weight_name=f"{prefix}.A_log",
            data_type=torch.float32,
            weight_shape=(num_heads,),
        )
        self.kda_dt_bias = TpParameterWeight(
            weight_name=f"{prefix}.dt_bias",
            data_type=torch.float32,
            weight_shape=(projection_size,),
        )
        if self.use_full_rank_gate:
            self.kda_g_proj = ROWMMWeight(
                in_dim=hidden_size,
                out_dims=[projection_size],
                weight_names=f"{prefix}.g_proj.weight",
                data_type=self.data_type_,
                quant_method=self.get_quant_method("g_proj"),
            )
        else:
            self.kda_g_a_proj = ROWMMWeight(
                in_dim=hidden_size,
                out_dims=[head_dim],
                weight_names=f"{prefix}.g_a_proj.weight",
                data_type=self.data_type_,
                quant_method=self.get_quant_method("g_a_proj"),
                tp_rank=0,
                tp_world_size=1,
            )
            self.kda_g_b_proj = ROWMMWeight(
                in_dim=head_dim,
                out_dims=[projection_size],
                weight_names=f"{prefix}.g_b_proj.weight",
                data_type=self.data_type_,
                quant_method=self.get_quant_method("g_b_proj"),
            )
        self.kda_o_norm = GatedRMSNormWeight(
            dim=head_dim,
            weight_name=f"{prefix}.o_norm.weight",
            data_type=self.data_type_,
            activation="sigmoid",
        )
        self.kda_o_proj = COLMMWeight(
            in_dim=projection_size,
            out_dims=[hidden_size],
            weight_names=f"{prefix}.o_proj.weight",
            data_type=self.data_type_,
            quant_method=self.get_quant_method("o_proj"),
        )

    def _init_mla_output_gate(self):
        prefix = f"model.layers.{self.layer_num_}.self_attn"
        self.mla_o_gate_proj = ROWMMWeight(
            in_dim=self.n_embed,
            out_dims=[self.o_in_dim],
            weight_names=f"{prefix}.g_proj.weight",
            data_type=self.data_type_,
            quant_method=self.get_quant_method("mla_o_gate_proj"),
        )

    def _init_moe(self):
        prefix = f"model.layers.{self.layer_num_}.block_sparse_moe"
        self.moe_gate = ROWMMWeight(
            in_dim=self.n_embed,
            out_dims=[self.n_routed_experts],
            weight_names=f"{prefix}.gate.weight",
            data_type=self.data_type_,
            quant_method=None,
            tp_rank=0,
            tp_world_size=1,
        )
        if self.num_fused_shared_experts == 0:
            self._init_shared_experts(f"{prefix}.shared_experts")
        expert_hidden_size = self.moe_latent_size or self.n_embed
        if self.moe_latent_size is not None:
            self.moe_latent_down_proj = ROWMMWeight(
                in_dim=self.n_embed,
                out_dims=[self.moe_latent_size],
                weight_names=f"{prefix}.routed_expert_down_proj.weight",
                data_type=self.data_type_,
                quant_method=self.get_quant_method("moe_latent_down_proj"),
                tp_rank=0,
                tp_world_size=1,
            )
            self.moe_latent_up_proj = ROWMMWeight(
                in_dim=self.moe_latent_size,
                out_dims=[self.n_embed],
                weight_names=f"{prefix}.routed_expert_up_proj.weight",
                data_type=self.data_type_,
                quant_method=self.get_quant_method("moe_latent_up_proj"),
                tp_rank=0,
                tp_world_size=1,
            )
            if self.latent_moe_use_norm:
                self.moe_latent_norm = RMSNormWeight(
                    dim=self.moe_latent_size,
                    weight_name=f"{prefix}.routed_expert_norm.weight",
                    data_type=self.data_type_,
                )
        self.experts = FusedMoeWeight(
            gate_proj_name="w1",
            down_proj_name="w2",
            up_proj_name="w3",
            e_score_correction_bias_name=f"{prefix}.gate.e_score_correction_bias",
            weight_prefix=f"{prefix}.experts",
            n_routed_experts=self.n_routed_experts,
            hidden_size=expert_hidden_size,
            moe_intermediate_size=self.moe_inter,
            data_type=self.data_type_,
            quant_method=self.quant_cfg.get_quant_method(self.layer_num_, "fused_moe"),
            num_fused_shared_experts=self.num_fused_shared_experts,
            layer_num=self.layer_num_,
            network_config=self.network_config_,
            activation=self.hidden_act,
            activation_situ_beta=self.activation_situ_beta,
            activation_situ_linear_beta=self.activation_situ_linear_beta,
        )

    def _init_shared_experts(self, prefix):
        shared_intermediate_size = self.moe_inter * self.network_config_["n_shared_experts"]
        enable_ep_moe = get_env_start_args().enable_ep_moe
        parallel_kwargs = {"tp_rank": 0, "tp_world_size": 1} if enable_ep_moe else {}
        self.gate_up_proj = ROWMMWeight(
            in_dim=self.n_embed,
            out_dims=[shared_intermediate_size, shared_intermediate_size],
            weight_names=[f"{prefix}.gate_proj.weight", f"{prefix}.up_proj.weight"],
            data_type=self.data_type_,
            quant_method=self.get_quant_method("gate_up_proj"),
            **parallel_kwargs,
        )
        self.down_proj = COLMMWeight(
            in_dim=shared_intermediate_size,
            out_dims=[self.n_embed],
            weight_names=f"{prefix}.down_proj.weight",
            data_type=self.data_type_,
            quant_method=self.get_quant_method("down_proj"),
            **parallel_kwargs,
        )

    def _rename_shared_experts(self, weights, weight_scale_suffix):
        prefix = f"model.layers.{self.layer_num_}.block_sparse_moe"
        expert_id = self.n_routed_experts
        for old_proj, new_proj in (
            ("gate_proj", "w1"),
            ("down_proj", "w2"),
            ("up_proj", "w3"),
        ):
            old_name = f"{prefix}.shared_experts.{old_proj}.weight"
            if old_name in weights:
                weights[f"{prefix}.experts.{expert_id}.{new_proj}.weight"] = weights[old_name]
            if self.quant_cfg.quantized_weight and weight_scale_suffix is not None:
                old_scale = f"{prefix}.shared_experts.{old_proj}.{weight_scale_suffix}"
                if old_scale in weights:
                    weights[f"{prefix}.experts.{expert_id}.{new_proj}.{weight_scale_suffix}"] = weights[old_scale]

    def _init_norm(self):
        prefix = f"model.layers.{self.layer_num_}"
        self.att_norm_weight_ = RMSNormWeight(
            dim=self.n_embed,
            weight_name=f"{prefix}.input_layernorm.weight",
            data_type=self.data_type_,
        )
        self.ffn_norm_weight_ = RMSNormWeight(
            dim=self.n_embed,
            weight_name=f"{prefix}.post_attention_layernorm.weight",
            data_type=self.data_type_,
        )
        if not self.is_linear_attention_layer:
            self.kv_a_layernorm_ = RMSNormWeight(
                dim=self.kv_lora_rank,
                weight_name=f"{prefix}.self_attn.kv_a_layernorm.weight",
                data_type=self.data_type_,
            )
            if self.q_lora_rank is not None:
                self.q_a_layernorm_ = RMSNormWeight(
                    dim=self.q_lora_rank,
                    weight_name=f"{prefix}.self_attn.q_a_layernorm.weight",
                    data_type=self.data_type_,
                )

    def load_hf_weights(self, weights):
        if self.attnres_config is not None:
            query_weights = [self.attnres_attn_query, self.attnres_mlp_query]
            if hasattr(self, "attnres_final_query"):
                query_weights.append(self.attnres_final_query)
            for query_weight in query_weights:
                name = query_weight.weight_name
                if name in weights:
                    weights[name] = normalize_attnres_query_weight(weights[name], self.n_embed)
        if self.is_linear_attention_layer:
            prefix = f"model.layers.{self.layer_num_}.self_attn"
            a_log_name = f"{prefix}.A_log"
            if a_log_name in weights:
                weights[a_log_name] = weights[a_log_name].reshape(-1)
            for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
                weight_name = f"{prefix}.{name}.weight"
                if weight_name in weights and weights[weight_name].ndim == 3:
                    weights[weight_name] = weights[weight_name].squeeze(1)
        super().load_hf_weights(weights)
