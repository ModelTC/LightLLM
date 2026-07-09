from lightllm.common.basemodel import PreAndPostLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import (
    EmbeddingWeight,
    LMHeadWeight,
    RMSNormWeight,
    ROWMMWeight,
)
from lightllm.common.quantization import Quantcfg


class Qwen3DFlashPreAndPostLayerWeight(PreAndPostLayerWeight):
    """Weights outside the DFlash decoder stack.

    DFlash checkpoints are stored as DSpark-family models, not as Qwen3 causal
    LM checkpoints.  Their top-level names are:

    - embed_tokens.weight: [vocab_size, hidden_size]
    - fc.weight: [hidden_size, hidden_size * len(target_layer_ids)]
    - hidden_norm.weight: [hidden_size]
    - norm.weight: [hidden_size]
    - lm_head.weight: [vocab_size, hidden_size]
    """

    def __init__(self, data_type, network_config, quant_cfg: Quantcfg):
        super().__init__(data_type, network_config)
        self.quant_cfg = quant_cfg

        hidden_size = network_config["hidden_size"]
        vocab_size = network_config["vocab_size"]
        target_layer_num = len(network_config["target_layer_ids"])

        self.wte_weight_ = EmbeddingWeight(
            dim=hidden_size,
            vocab_size=vocab_size,
            weight_name="embed_tokens.weight",
            data_type=self.data_type_,
        )
        self.fc_weight_ = ROWMMWeight(
            in_dim=hidden_size * target_layer_num,
            out_dims=[hidden_size],
            weight_names="fc.weight",
            data_type=self.data_type_,
            quant_method=self.quant_cfg.get_quant_method(0, "fc"),
            tp_rank=0,
            tp_world_size=1,
        )
        self.hidden_norm_weight_ = RMSNormWeight(
            dim=hidden_size,
            weight_name="hidden_norm.weight",
            data_type=self.data_type_,
        )
        self.final_norm_weight_ = RMSNormWeight(
            dim=hidden_size,
            weight_name="norm.weight",
            data_type=self.data_type_,
        )
        self.lm_head_weight_ = LMHeadWeight(
            dim=hidden_size,
            vocab_size=vocab_size,
            weight_name="lm_head.weight",
            data_type=self.data_type_,
        )
        return
