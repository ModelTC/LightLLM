import torch
import math
import numpy as np
from lightllm.common.basemodel import TransformerLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import ROWMMWeight, COLMMWeight, RMSNormWeight, KVROWNMMWeight


class LlamaTransformerLayerWeight(TransformerLayerWeight):
    def __init__(
        self,
        layer_num,
        data_type,
        network_config,
        quant_cfg=None,
    ):
        super().__init__(layer_num, data_type, network_config, quant_cfg)
        return

    def _init_weight(self):
        self._init_qkv()
        self._init_o()
        self._init_ffn()
        self._init_norm()

    def _parse_config(self):
        self.n_head = self.network_config_["num_attention_heads"]
        self.q_head_num_ = self.network_config_["num_attention_heads"]
        self.k_head_num_ = self.network_config_["num_key_value_heads"]
        self.v_head_num_ = self.k_head_num_
        self.o_head_num_ = self.q_head_num_
        head_dim = self.network_config_["hidden_size"] // self.network_config_["num_attention_heads"]
        self.head_dim = self.network_config_.get("head_dim", head_dim)
        self.n_embed = self.network_config_["hidden_size"]
        self.n_inter = self.network_config_["intermediate_size"]

    def _init_weight_names(self):
        self._q_weight_name = f"model.layers.{self.layer_num_}.self_attn.q_proj.weight"
        self._q_bias_name = None
        self._k_weight_name = f"model.layers.{self.layer_num_}.self_attn.k_proj.weight"
        self._k_bias_name = None
        self._v_weight_name = f"model.layers.{self.layer_num_}.self_attn.v_proj.weight"
        self._v_bias_name = None
        self._kv_weight_name = f"model.layers.{self.layer_num_}.self_attn.kv_proj.weight"
        self._kv_bias_name = None
        self._o_weight_name = f"model.layers.{self.layer_num_}.self_attn.o_proj.weight"
        self._o_bias_name = None

        self._gate_weight_name = f"model.layers.{self.layer_num_}.mlp.gate_proj.weight"
        self._gate_bias_name = None
        self._up_weight_name = f"model.layers.{self.layer_num_}.mlp.up_proj.weight"
        self._up_bias_name = None
        self._gate_up_weight_name = f"model.layers.{self.layer_num_}.mlp.gate_up_proj.weight"
        self._gate_up_bias_name = None
        self._down_weight_name = f"model.layers.{self.layer_num_}.mlp.down_proj.weight"
        self._down_bias_name = None

        self._att_norm_weight_name = f"model.layers.{self.layer_num_}.input_layernorm.weight"
        self._att_norm_bias_name = None
        self._ffn_norm_weight_name = f"model.layers.{self.layer_num_}.post_attention_layernorm.weight"
        self._ffn_norm_bias_name = None

    def _init_qkv(self):
        in_dim = self.n_embed
        q_out_dim = self.q_head_num_ * self.head_dim
        self.q_proj = ROWMMWeight(
            in_dim=in_dim,
            out_dims=[q_out_dim],
            weight_names=self._q_weight_name,
            data_type=self.data_type_,
            bias_names=self._q_bias_name,
            quant_method=self.get_quant_method("q_proj"),
        )
        self.kv_proj = KVROWNMMWeight(
            in_dim=in_dim,
            kv_head_num=self.k_head_num_,
            head_dim=self.head_dim,
            weight_names=[self._k_weight_name, self._v_weight_name],
            data_type=self.data_type_,
            bias_names=[self._k_bias_name, self._v_bias_name],
            quant_method=self.get_quant_method("kv_proj"),
        )

    def _init_o(self):
        in_dim = self.o_head_num_ * self.head_dim
        out_dim = self.n_embed
        self.o_proj = COLMMWeight(
            in_dim=in_dim,
            out_dims=[out_dim],
            weight_names=self._o_weight_name,
            data_type=self.data_type_,
            bias_names=self._o_bias_name,
            quant_method=self.get_quant_method("o_proj"),
        )

    def _init_ffn(self):
        in_dim = self.n_embed
        out_dim = self.n_inter // self.tp_world_size_
        self.gate_up_proj = ROWMMWeight(
            in_dim=in_dim,
            out_dims=[out_dim, out_dim],
            weight_names=[self._gate_weight_name, self._up_weight_name],
            data_type=self.data_type_,
            bias_names=[self._gate_bias_name, self._up_bias_name],
            quant_method=self.get_quant_method("gate_up_proj"),
        )
        self.down_proj = COLMMWeight(
            in_dim=out_dim,
            out_dims=[in_dim],
            weight_names=self._down_weight_name,
            data_type=self.data_type_,
            bias_names=self._down_bias_name,
            quant_method=self.get_quant_method("down_proj"),
        )

    def _init_norm(self):
        hidden_size = self.network_config_["hidden_size"]
        self.att_norm_weight_ = RMSNormWeight(
            dim=hidden_size,
            weight_name=self._att_norm_weight_name,
            data_type=self.data_type_,
            bias_name=self._att_norm_bias_name,
        )
        self.ffn_norm_weight_ = RMSNormWeight(
            dim=hidden_size,
            weight_name=self._ffn_norm_weight_name,
            data_type=self.data_type_,
            bias_name=self._ffn_norm_bias_name,
        )
