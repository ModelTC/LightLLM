from lightllm.common.basemodel.layer_weights.meta_weights import (
    COLMMWeight,
    FusedMoeWeight,
    ROWMMWeight,
    QKVROWNMMWeight,
)
from lightllm.models.qwen3_5_moe.layer_weights.transformer_layer_weight import (
    Qwen35MOETransformerLayerWeight,
)
from lightllm.utils.envs_utils import get_env_start_args


class Qwen3_5MoeMTPTransformerLayerWeight(Qwen35MOETransformerLayerWeight):
    # MTP draft-model weights live under the `mtp.layers.*` checkpoint namespace; the
    # main-model attention/norm names (`model.layers.*`) are retargeted to it, while the
    # MoE expert / shared-expert names are built directly with the mtp prefix below.

    _MAIN_PREFIX = "model.layers."
    _MTP_PREFIX = "mtp.layers."

    _ATTN_NORM_NAME_ATTRS = (
        "_q_weight_name",
        "_q_norm_name",
        "_q_bias_name",
        "_k_weight_name",
        "_k_norm_name",
        "_k_bias_name",
        "_v_weight_name",
        "_v_bias_name",
        "_kv_weight_name",
        "_kv_bias_name",
        "_o_weight_name",
        "_o_bias_name",
        "_att_norm_weight_name",
        "_att_norm_bias_name",
        "_ffn_norm_weight_name",
        "_ffn_norm_bias_name",
    )

    def _retarget(self, name):
        if name is None:
            return None
        return name.replace(self._MAIN_PREFIX, self._MTP_PREFIX, 1)

    def _retarget_attn_norm_names(self):
        for attr in self._ATTN_NORM_NAME_ATTRS:
            setattr(self, attr, self._retarget(getattr(self, attr)))

    def _init_qkv(self):
        in_dim = self.n_embed
        q_out_dim = self.q_head_num_ * self.head_dim
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
        self._o_gate_weight_name = f"{self._MTP_PREFIX}{self.layer_num_}.self_attn.o_gate_proj.weight"
        self._o_gate_proj = ROWMMWeight(
            in_dim=in_dim,
            out_dims=[q_out_dim],
            weight_names=[self._o_gate_weight_name],
            data_type=self.data_type_,
            bias_names=None,
            quant_method=self.get_quant_method("o_gate_proj"),
        )

    def _init_weight_names(self):
        super()._init_weight_names()
        self._retarget_attn_norm_names()

    def _init_moe(self):
        moe_intermediate_size = self.network_config_["moe_intermediate_size"]
        self.moe_gate = ROWMMWeight(
            in_dim=self.network_config_["hidden_size"],
            out_dims=[self.n_routed_experts],
            weight_names=f"{self._MTP_PREFIX}{self.layer_num_}.mlp.gate.weight",
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
            weight_prefix=f"{self._MTP_PREFIX}{self.layer_num_}.mlp.experts",
            n_routed_experts=self.n_routed_experts,
            hidden_size=self.network_config_["hidden_size"],
            moe_intermediate_size=moe_intermediate_size,
            data_type=self.data_type_,
            quant_method=self.quant_cfg.get_quant_method(self.layer_num_, "fused_moe"),
            layer_num=self.layer_num_,
            network_config=self.network_config_,
        )
        self._init_gated_ffn()

    def _init_gated_ffn(self):
        hidden_size = self.network_config_["hidden_size"]
        if "shared_expert_intermediate_size" not in self.network_config_:
            return

        prefix = f"{self._MTP_PREFIX}{self.layer_num_}.mlp.shared_expert"
        inter_size = self.network_config_["shared_expert_intermediate_size"]
        if get_env_start_args().enable_ep_moe:
            self.gate_up_proj = ROWMMWeight(
                in_dim=hidden_size,
                out_dims=[inter_size, inter_size],
                weight_names=[f"{prefix}.gate_proj.weight", f"{prefix}.up_proj.weight"],
                data_type=self.data_type_,
                quant_method=self.get_quant_method("gate_up_proj"),
                tp_rank=0,
                tp_world_size=1,
            )
            self.down_proj = COLMMWeight(
                in_dim=inter_size,
                out_dims=[hidden_size],
                weight_names=f"{prefix}.down_proj.weight",
                data_type=self.data_type_,
                quant_method=self.get_quant_method("down_proj"),
                tp_rank=0,
                tp_world_size=1,
            )
        else:
            self.gate_up_proj = ROWMMWeight(
                in_dim=hidden_size,
                out_dims=[inter_size, inter_size],
                weight_names=[f"{prefix}.gate_proj.weight", f"{prefix}.up_proj.weight"],
                data_type=self.data_type_,
                quant_method=self.get_quant_method("gate_up_proj"),
            )
            self.down_proj = COLMMWeight(
                in_dim=inter_size,
                out_dims=[hidden_size],
                weight_names=f"{prefix}.down_proj.weight",
                data_type=self.data_type_,
                quant_method=self.get_quant_method("down_proj"),
            )

        self.ffn_gate = ROWMMWeight(
            in_dim=hidden_size,
            out_dims=[1],
            weight_names=f"{self._MTP_PREFIX}{self.layer_num_}.mlp.shared_expert_gate.weight",
            data_type=self.data_type_,
            bias_names=None,
            quant_method=None,
            tp_rank=0,
            tp_world_size=1,
        )
