from lightllm.common.basemodel import PreAndPostLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import (
    EmbeddingWeight,
    LMHeadWeight,
    ParameterWeight,
    RMSNormWeight,
    ROWMMWeight,
)
from lightllm.common.quantization import Quantcfg


class Qwen3DFlash2PreAndPostLayerWeight(PreAndPostLayerWeight):
    """DFlash2 projection, normalization, and candidate-selector weights."""

    def __init__(self, data_type, network_config, quant_cfg: Quantcfg):
        super().__init__(data_type, network_config)
        self.quant_cfg = quant_cfg

        hidden_size = network_config["hidden_size"]
        vocab_size = network_config["vocab_size"]
        target_layer_num = len(network_config["target_layer_ids"])
        selector_rank = int(network_config["selector_rank"])

        # Published DFlash2 checkpoints share these two large weights with the
        # target model. Qwen3DFlash2Model wires them up before loading weights.
        self.wte_weight_: EmbeddingWeight = None
        self.lm_head_weight_: LMHeadWeight = None
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
        self.selector_hidden_projection_weight_ = ROWMMWeight(
            in_dim=hidden_size,
            out_dims=[selector_rank],
            weight_names="candidate_selector.hidden_projection.weight",
            data_type=self.data_type_,
            quant_method=None,
            tp_rank=0,
            tp_world_size=1,
        )
        self.selector_predecessor_codebook_weight_ = ParameterWeight(
            weight_name="candidate_selector.predecessor_codebook",
            data_type=self.data_type_,
            weight_shape=(vocab_size, selector_rank),
        )
        self.selector_successor_codebook_weight_ = ParameterWeight(
            weight_name="candidate_selector.successor_codebook",
            data_type=self.data_type_,
            weight_shape=(vocab_size, selector_rank),
        )
