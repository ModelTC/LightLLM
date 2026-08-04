from lightllm.common.basemodel.layer_weights.meta_weights import ParameterWeight
from lightllm.common.quantization import Quantcfg
from lightllm.models.qwen3_dflash.layer_weights.pre_and_post_layer_weight import Qwen3DFlashPreAndPostLayerWeight


class Qwen3DSparkPreAndPostLayerWeight(Qwen3DFlashPreAndPostLayerWeight):
    """DSpark heads on top of the shared DFlash block backbone weights."""

    def __init__(self, data_type, network_config, quant_cfg: Quantcfg):
        super().__init__(data_type, network_config, quant_cfg)

        hidden_size = network_config["hidden_size"]
        vocab_size = network_config["vocab_size"]
        markov_rank = int(network_config.get("markov_rank", 0))
        enable_confidence_head = bool(network_config.get("enable_confidence_head", False))
        confidence_head_with_markov = bool(network_config.get("confidence_head_with_markov", False))
        assert (
            not confidence_head_with_markov or markov_rank > 0
        ), "confidence_head_with_markov requires markov_rank > 0"

        self.markov_w1_weight_ = None
        self.markov_w2_weight_ = None
        self.markov_gate_proj_weight_ = None
        self.markov_joint_proj_weight_ = None
        self.markov_rank = markov_rank
        self.markov_head_type = str(network_config.get("markov_head_type", "")).lower()
        if markov_rank > 0:
            self.markov_w1_weight_ = ParameterWeight(
                weight_name="markov_head.markov_w1.weight",
                data_type=self.data_type_,
                weight_shape=(vocab_size, markov_rank),
            )
            self.markov_w2_weight_ = ParameterWeight(
                weight_name="markov_head.markov_w2.weight",
                data_type=self.data_type_,
                weight_shape=(vocab_size, markov_rank),
            )
            if self.markov_head_type == "gated":
                self.markov_gate_proj_weight_ = ParameterWeight(
                    weight_name="markov_head.gate_proj.weight",
                    bias_name="markov_head.gate_proj.bias",
                    data_type=self.data_type_,
                    weight_shape=(markov_rank, hidden_size + markov_rank),
                    bias_shape=(markov_rank,),
                )
            elif self.markov_head_type == "rnn":
                self.markov_joint_proj_weight_ = ParameterWeight(
                    weight_name="markov_head.joint_proj.weight",
                    bias_name="markov_head.joint_proj.bias",
                    data_type=self.data_type_,
                    weight_shape=(3 * markov_rank, hidden_size + 2 * markov_rank),
                    bias_shape=(3 * markov_rank,),
                )
            else:
                assert self.markov_head_type == "vanilla", f"unsupported DSpark markov head {self.markov_head_type}"

        self.confidence_head_weight_ = None
        if enable_confidence_head:
            confidence_input_dim = hidden_size + (markov_rank if confidence_head_with_markov else 0)
            self.confidence_head_weight_ = ParameterWeight(
                weight_name="confidence_head.proj.weight",
                bias_name="confidence_head.proj.bias",
                data_type=self.data_type_,
                weight_shape=(1, confidence_input_dim),
                bias_shape=(1,),
            )
        return
