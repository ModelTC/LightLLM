from lightllm.common.basemodel.layer_weights.pre_and_post_layer_weight import PreAndPostLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import NoTpGEMMANormWeight
from lightllm.common.quantization import Quantcfg
from lightllm.models.qwen3_5.layer_weights.pre_and_post_layer_weight import Qwen35PreAndPostLayerWeight
from .awq_weight import LoopAWQLMHeadWeight, LoopINT4EmbeddingWeight


class SenseNovaLoopPreAndPostLayerWeight(Qwen35PreAndPostLayerWeight):
    def __init__(self, data_type, network_config):
        quantization = network_config.get("quantization_config") or {}
        embedding_awq = quantization.get("embedding")

        if quantization.get("quant_method") != "awq" or not quantization.get("lm_head") or embedding_awq is None:
            super().__init__(data_type, network_config)
            return

        embedding_config = {**quantization, **embedding_awq}
        if (
            embedding_config.get("bits"),
            embedding_config.get("layout", "row_major"),
            embedding_config.get("zero_point"),
        ) != (4, "row_major", True):
            raise ValueError("SenseNova Loop embedding requires row_major INT4 with zero points")

        PreAndPostLayerWeight.__init__(self, data_type, network_config)
        if network_config.get("tie_word_embeddings", False):
            raise ValueError("SenseNova AWQ requires separately exported embedding and lm_head tensors")
        dim, vocab = network_config["hidden_size"], network_config["vocab_size"]
        self.wte_weight_ = LoopINT4EmbeddingWeight(dim, vocab, embedding_config["group_size"], data_type)
        self.lm_head_weight_ = LoopAWQLMHeadWeight(
            dim, vocab, data_type, Quantcfg(network_config).get_quant_method(0, "lm_head")
        )
        self.final_norm_weight_ = NoTpGEMMANormWeight(dim=dim, weight_name="model.norm.weight", data_type=data_type)
