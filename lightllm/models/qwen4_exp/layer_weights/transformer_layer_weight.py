import torch

from lightllm.models.qwen3_5_moe.layer_weights.transformer_layer_weight import Qwen35MOETransformerLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import QKGEMMANormWeight
from lightllm.common.quantization.scalar_projection import StableScalarProjection
from lightllm.common.quantization.aligned_projection import OutputAlignedProjection
from lightllm.common.quantization.no_quant import NoQuantization
from lightllm.models.qwen4_exp.checkpoint import normalize_text_weight_names, ngram_table_shape
from .components import gated_residual_weight, qsa_indexer_weight, ple_weight, ngram_hasher_weight
from .norm_weight import Qwen4ExpGatedNormWeight
from .embedding_weight import ngram_embedding_weight


class Qwen4ExpTransformerLayerWeight(Qwen35MOETransformerLayerWeight):
    def get_quant_method(self, name):
        method = super().get_quant_method(name)
        if name == "in_proj_weight" and type(method) is NoQuantization:
            aligned = OutputAlignedProjection()
            aligned.hf_quantization_config = method.hf_quantization_config
            return aligned
        return method

    def _init_moe(self):
        super()._init_moe()

        self.shared_expert_gate.quant_method = StableScalarProjection()

    def _init_weight(self):
        config = self.network_config_
        layer_types = config.get("layer_types")
        if layer_types is not None:
            layer_type = layer_types[self.layer_num_]
            if layer_type not in ("linear_attention", "full_attention", "indexed_attention"):
                raise ValueError(f"Unsupported Qwen4Exp layer type {layer_type}")
            self.is_linear_attention_layer = layer_type == "linear_attention"
        super()._init_weight()
        prefix = f"model.layers.{self.layer_num_}"
        self.attn_hyper_connection = gated_residual_weight(f"{prefix}.attn_hyper_connection", config, self.data_type_)
        self.mlp_hyper_connection = gated_residual_weight(f"{prefix}.mlp_hyper_connection", config, self.data_type_)
        if not self.is_linear_attention_layer:
            self.indexer = qsa_indexer_weight(f"{prefix}.self_attn.indexer", config, self.data_type_)
        ple_layers = sorted(set(config.get("ple_layer_ids", [])))
        if self.layer_num_ + 1 in ple_layers:
            if not self.is_linear_attention_layer:
                raise ValueError("Qwen4Exp PLE must be attached to a linear-attention layer")
            self.ple = ple_weight(f"{prefix}.ple", config, self.data_type_)
            self.ngram_hasher = ngram_hasher_weight(f"{prefix}.ple.ple_embedding", config)
            heads = (config["ngram_size"] - 1) * config["heads_per_ngram"]
            rows, dim = ngram_table_shape(
                config["ngram_vocab_size_base"],
                heads,
                ple_layers.index(self.layer_num_ + 1),
                config.get("ple_embed_dim", config["hidden_size"]),
                config["make_ngram_vocab_size_divisible_by"],
            )
            self.ngram_embedding = ngram_embedding_weight(
                dim,
                rows,
                f"{prefix}.ple.ple_embedding.ngram_embedding.weight",
                self.data_type_,
                config["split_ngram_parts"],
            )

    def _init_gdn_weight(self):
        super()._init_gdn_weight()
        config = self.network_config_
        self.linear_norm = Qwen4ExpGatedNormWeight(
            self.linear_v_head_dim,
            f"model.layers.{self.layer_num_}.linear_attn.norm.weight",
            self.data_type_,
            config.get("output_gate_type") or config["hidden_act"],
        )

    def _init_norm(self):
        if not self.is_linear_attention_layer:
            self.qk_norm_weight_ = QKGEMMANormWeight(
                self.head_dim,
                self._q_norm_name,
                self._k_norm_name,
                torch.float32,
            )

    def load_hf_weights(self, weights):
        normalize_text_weight_names(weights)
        super().load_hf_weights(weights)
