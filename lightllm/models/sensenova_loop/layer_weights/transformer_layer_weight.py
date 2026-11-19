import torch

from lightllm.common.basemodel.layer_weights.meta_weights import (
    ParameterWeight,
    ROWMMWeight,
    TpParameterWeight,
    GatedRMSNormWeight,
)
from lightllm.models.qwen3_5.layer_weights.transformer_layer_weight import Qwen35TransformerLayerWeight
from .awq_weight import InputScaledAWQBundle, InputScaledAWQWeight


class SenseNovaLoopTransformerLayerWeight(Qwen35TransformerLayerWeight):
    @property
    def runtime_scale_awq(self):
        return self.network_config_.get("loop_awq_config") is not None

    def _scaled_projection(self, in_dim, out_dim, name, quant_name):
        return InputScaledAWQWeight(in_dim, out_dim, name, self.data_type_, self.get_quant_method(quant_name))

    def _scaled_bundle(self, in_dim, out_dims, names, quant_name):
        return InputScaledAWQBundle(in_dim, out_dims, names, self.data_type_, lambda: self.get_quant_method(quant_name))

    def _init_qkv(self):
        if not self.runtime_scale_awq:
            return super()._init_qkv()
        self._o_gate_weight_name = f"model.layers.{self.layer_num_}.self_attn.o_gate_proj.weight"
        self.qkvo_gate_proj = self._scaled_bundle(
            self.n_embed,
            [
                self.q_head_num_ * self.head_dim,
                self.k_head_num_ * self.head_dim,
                self.v_head_num_ * self.head_dim,
                self.q_head_num_ * self.head_dim,
            ],
            [self._q_weight_name, self._k_weight_name, self._v_weight_name, self._o_gate_weight_name],
            "qkv_proj",
        )

    def _init_o(self):
        if not self.runtime_scale_awq:
            return super()._init_o()
        self.o_proj = self._scaled_projection(
            self.o_head_num_ * self.head_dim, self.n_embed, self._o_weight_name, "o_proj"
        )

    def _init_ffn(self):
        if not self.runtime_scale_awq:
            return super()._init_ffn()
        self.gate_up_proj = self._scaled_bundle(
            self.n_embed,
            [self.n_inter, self.n_inter],
            [self._gate_weight_name, self._up_weight_name],
            "gate_up_proj",
        )
        self.down_proj = self._scaled_projection(self.n_inter, self.n_embed, self._down_weight_name, "down_proj")

    def _init_gdn_weight(self):
        if not self.runtime_scale_awq:
            return super()._init_gdn_weight()
        prefix = f"model.layers.{self.layer_num_}.linear_attn"
        qk_dim = self.linear_num_k_heads * self.linear_k_head_dim
        v_dim = self.linear_num_v_heads * self.linear_v_head_dim
        self.linear_conv1d = ROWMMWeight(
            in_dim=self.network_config_.get("linear_conv_kernel_dim", 4),
            out_dims=[2 * qk_dim + v_dim],
            weight_names=f"{prefix}.conv1d.weight",
            data_type=self.data_type_,
        )
        self.linear_in_proj = self._scaled_bundle(
            self.n_embed,
            [qk_dim, qk_dim, v_dim, v_dim],
            [f"{prefix}.in_proj_{part}.weight" for part in ("q", "k", "v", "z")],
            "in_proj_weight",
        )
        self.linear_gate_proj = ROWMMWeight(
            in_dim=self.n_embed,
            out_dims=[self.linear_num_v_heads] * 2,
            weight_names=[f"{prefix}.in_proj_{part}.weight" for part in ("b", "a")],
            data_type=self.data_type_,
        )
        self.linear_out_proj = self._scaled_projection(
            v_dim, self.n_embed, f"{prefix}.out_proj.weight", "out_proj_weight"
        )
        self.linear_dt_bias = TpParameterWeight(
            weight_name=f"{prefix}.dt_bias",
            data_type=torch.float32,
            bias_name=None,
            weight_shape=(self.linear_num_v_heads,),
            bias_shape=None,
        )
        self.linear_A_log = TpParameterWeight(
            weight_name=f"{prefix}.A_log",
            data_type=torch.float32,
            bias_name=None,
            weight_shape=(self.linear_num_v_heads,),
            bias_shape=None,
        )
        self.linear_norm = GatedRMSNormWeight(
            dim=self.linear_v_head_dim,
            weight_name=f"{prefix}.norm.weight",
            data_type=self.data_type_,
        )

    def _split_q_with_gate(self, weights):
        super()._split_q_with_gate(weights)
        if self.runtime_scale_awq and not self.is_linear_attention_layer:
            name = self._q_weight_name.removesuffix(".weight") + ".input_scale"
            if name in weights:
                weights[self._o_gate_weight_name.removesuffix(".weight") + ".input_scale"] = weights[name]

    def _split_linear_in_proj_qkv(self, weights):
        super()._split_linear_in_proj_qkv(weights)
        if self.runtime_scale_awq:
            prefix = f"model.layers.{self.layer_num_}.linear_attn"
            scale = weights.get(f"{prefix}.in_proj_qkv.input_scale")
            if scale is not None:
                for part in ("q", "k", "v"):
                    weights[f"{prefix}.in_proj_{part}.input_scale"] = scale

    def _init_weight(self):
        execution_index = self.network_config_["loop_physical_indices"].index(self.layer_num_)
        self.is_linear_attention_layer = self.network_config_["layer_types"][execution_index] == "linear_attention"
        super()._init_weight()
        self.mhc = self.network_config_.get("mhc")
        if self.mhc is not None and self.mhc["start"] <= self.layer_num_ < self.mhc["end"]:
            self._init_mhc()

    def _init_gdn_in_proj(self, out_dims, weight_names, quant_method):
        if quant_method.method_name in ("awq", "awq_marlin"):
            quantization = quant_method.hf_quantization_config or {}
            excluded = {
                name.replace("model.language_model.", "model.", 1)
                for name in (quantization.get("modules_to_not_convert") or ())
            }
            gate_weights = weight_names[-2:]
            gate_modules = [name.removesuffix(".weight") for name in gate_weights]
            if all(any(module == name or module.endswith(f".{name}") for name in excluded) for module in gate_modules):
                self.linear_gate_proj = ROWMMWeight(
                    in_dim=self.network_config_["hidden_size"],
                    out_dims=out_dims[-2:],
                    weight_names=gate_weights,
                    data_type=self.data_type_,
                )
                out_dims, weight_names = out_dims[:-2], weight_names[:-2]

        super()._init_gdn_in_proj(out_dims, weight_names, quant_method)

    def _init_mhc(self):
        prefix = f"model.layers.{self.layer_num_}"
        streams = self.mhc["streams"]
        mix_size = (2 + streams) * streams
        flattened_hidden = streams * self.n_embed
        self.mhc_attn = ParameterWeight(
            weight_name=f"{prefix}.mhc_attn.weight",
            bias_name=f"{prefix}.mhc_attn.bias",
            data_type=torch.float32,
            weight_shape=(mix_size, flattened_hidden),
            bias_shape=(mix_size,),
        )
        self.mhc_ffn = ParameterWeight(
            weight_name=f"{prefix}.mhc_ffn.weight",
            bias_name=f"{prefix}.mhc_ffn.bias",
            data_type=torch.float32,
            weight_shape=(mix_size, flattened_hidden),
            bias_shape=(mix_size,),
        )
