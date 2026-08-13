from lightllm.common.basemodel.layer_weights.meta_weights import EmbeddingWeight, LMHeadWeight, ROWMMWeight
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
            # W1 is read once per Markov step; replication avoids a sequential TP collective on every lookup.
            self.markov_w1_weight_ = EmbeddingWeight(
                dim=markov_rank,
                vocab_size=vocab_size,
                weight_name="markov_head.markov_w1.weight",
                data_type=self.data_type_,
                tp_rank=0,
                tp_world_size=1,
            )
            self.markov_w2_weight_ = LMHeadWeight(
                dim=markov_rank,
                vocab_size=vocab_size,
                weight_name="markov_head.markov_w2.weight",
                data_type=self.data_type_,
            )
            if self.markov_head_type == "gated":
                self.markov_gate_proj_weight_ = ROWMMWeight(
                    in_dim=hidden_size + markov_rank,
                    out_dims=[markov_rank],
                    weight_names="markov_head.gate_proj.weight",
                    bias_names="markov_head.gate_proj.bias",
                    data_type=self.data_type_,
                    # W8A8 MM does not support the Markov projection bias.
                    quant_method=None,
                    tp_rank=0,
                    tp_world_size=1,
                )
            elif self.markov_head_type == "rnn":
                self.markov_joint_proj_weight_ = ROWMMWeight(
                    in_dim=hidden_size + 2 * markov_rank,
                    out_dims=[3 * markov_rank],
                    weight_names="markov_head.joint_proj.weight",
                    bias_names="markov_head.joint_proj.bias",
                    data_type=self.data_type_,
                    quant_method=None,
                    tp_rank=0,
                    tp_world_size=1,
                )
            else:
                assert self.markov_head_type == "vanilla", f"unsupported DSpark markov head {self.markov_head_type}"

        self.confidence_head_weight_ = None
        if enable_confidence_head:
            confidence_input_dim = hidden_size + (markov_rank if confidence_head_with_markov else 0)
            self.confidence_head_weight_ = ROWMMWeight(
                in_dim=confidence_input_dim,
                out_dims=[1],
                weight_names="confidence_head.proj.weight",
                bias_names="confidence_head.proj.bias",
                data_type=self.data_type_,
                # The confidence head has a bias and only one output channel.
                # Keep it in model dtype because W8A8 MM does not support bias.
                quant_method=None,
                tp_rank=0,
                tp_world_size=1,
            )
