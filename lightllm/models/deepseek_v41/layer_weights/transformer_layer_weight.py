from lightllm.common.basemodel.layer_weights.meta_weights import (
    ROWMMWeight,
    RMSNormWeight,
    ParameterWeight,
)
from lightllm.models.deepseek_v4.layer_weights.transformer_layer_weight import DeepseekV4TransformerLayerWeight
from ..engram import EngramEmbeddingWeight


class DeepseekV41TransformerLayerWeight(DeepseekV4TransformerLayerWeight):
    def _parse_config(self):
        super()._parse_config()
        self.has_compressor = self.layer_num_ in self.network_config_["kv_source_layer_ids"]
        self.has_indexer = self.layer_num_ in self.network_config_["index_source_layer_ids"]
        self.has_engram = self.layer_num_ in self.network_config_["engram_layer_ids"]

    def _init_weight(self):
        super()._init_weight()
        if self.has_engram:
            cfg = self.network_config_
            prefix = f"{self.prefix}.engram"
            index = cfg["engram_layer_ids"].index(self.layer_num_)
            self.engram_embed_ = EngramEmbeddingWeight(
                f"{prefix}.embed",
                cfg["engram_num_embeddings"][index],
                cfg["engram_head_dim"],
            )
            self.engram_wkv_ = ROWMMWeight(
                in_dim=(cfg["engram_max_ngram_size"] - 1) * cfg["engram_n_heads"] * cfg["engram_head_dim"],
                out_dims=[(self.hc_mult + 1) * self.hidden],
                weight_names=f"{prefix}.wkv.weight",
                data_type=self.data_type_,
                quant_method=self.get_quant_method("engram_wkv"),
                tp_rank=0,
                tp_world_size=1,
            )
            self.engram_q_ = ParameterWeight(f"{prefix}.q_weight", self.data_type_, (self.hc_mult, self.hidden))
            self.engram_k_ = ParameterWeight(f"{prefix}.k_weight", self.data_type_, (self.hc_mult, self.hidden))

    def _init_compressor(self):
        prefix = f"{self.prefix}.attn.compressor"
        names = [f"{prefix}.wkv.weight"]
        if self.compress_ratio == 2:
            names.append(f"{prefix}.wgate.weight")
        self.compressor_wkv_gate_ = ROWMMWeight(
            in_dim=self.hidden,
            out_dims=[self.head_dim] * len(names),
            weight_names=names,
            data_type=self.data_type_,
            quant_method=None,
            tp_rank=0,
            tp_world_size=1,
        )
        self.compressor_norm_ = RMSNormWeight(self.head_dim, f"{prefix}.norm.weight", self.data_type_)

    def _init_indexer_k(self):
        p = f"{self.prefix}.attn.indexer"
        if self.has_compressor:
            self.idx_wk_ = ROWMMWeight(
                in_dim=self.head_dim,
                out_dims=[self.index_head_dim],
                weight_names=f"{p}.wk.weight",
                data_type=self.data_type_,
                quant_method=None,
                tp_rank=0,
                tp_world_size=1,
            )
            self.idx_k_norm_ = RMSNormWeight(self.index_head_dim, f"{p}.k_norm.weight", self.data_type_)
