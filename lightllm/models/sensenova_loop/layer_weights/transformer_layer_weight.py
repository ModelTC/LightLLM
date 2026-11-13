import torch

from lightllm.common.basemodel.layer_weights.meta_weights import ParameterWeight, ROWMMWeight
from lightllm.models.qwen3_5.layer_weights.transformer_layer_weight import Qwen35TransformerLayerWeight


class SenseNovaLoopTransformerLayerWeight(Qwen35TransformerLayerWeight):
    def _init_weight(self):
        execution_index = self.network_config_["loop_physical_indices"].index(self.layer_num_)
        self.is_linear_attention_layer = self.network_config_["layer_types"][execution_index] == "linear_attention"
        super()._init_weight()
        self.dsa = self.network_config_.get("dsa")
        if self.dsa and not self.is_linear_attention_layer:
            self._init_dsa()
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

    def _init_dsa(self):
        prefix = f"model.layers.{self.layer_num_}.self_attn.indexer"
        heads, dim = self.dsa["num_heads"], self.dsa["head_dim"]
        self.dsa_wq = ParameterWeight(f"{prefix}.wq.weight", self.data_type_, (heads * dim, self.n_embed))
        self.dsa_wk = ParameterWeight(f"{prefix}.wk.weight", self.data_type_, (dim, self.n_embed))
        self.dsa_weights = ParameterWeight(f"{prefix}.weights_proj.weight", self.data_type_, (heads, self.n_embed))
        self.dsa_norm = ParameterWeight(
            f"{prefix}.k_norm.weight",
            self.data_type_,
            (dim,),
            bias_name=f"{prefix}.k_norm.bias",
            bias_shape=(dim,),
        )

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
