import torch
from lightllm.common.basemodel import PreAndPostLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import (
    EmbeddingWeight,
    LMHeadWeight,
    ParameterWeight,
    RMSNormWeight,
    ROWMMWeight,
)
from lightllm.common.quantization import Quantcfg


class Qwen3EaglePreAndPostLayerWeight(PreAndPostLayerWeight):
    def __init__(self, data_type, network_config, quant_cfg: Quantcfg):
        super().__init__(data_type, network_config)
        self.quant_cfg: Quantcfg = quant_cfg
        hidden_size = network_config["hidden_size"]
        target_layer_num = len(network_config.get("target_layer_ids", [0, 1, 2]))
        draft_vocab_size = network_config.get("draft_vocab_size")
        target_vocab_size = network_config.get("target_vocab_size")
        vocab_size = draft_vocab_size if draft_vocab_size is not None else network_config["vocab_size"]
        self.fc_weight_ = ROWMMWeight(
            in_dim=hidden_size * target_layer_num,
            out_dims=[hidden_size],
            weight_names="fc.weight",
            quant_method=self.quant_cfg.get_quant_method(0, "fc"),
            data_type=self.data_type_,
            tp_rank=0,
            tp_world_size=1,
        )
        self.final_norm_weight_: RMSNormWeight = RMSNormWeight(
            dim=hidden_size,
            weight_name="norm.weight",
            data_type=self.data_type_,
        )

        self.d2t_weight_ = None
        self.t2d_weight_ = None
        if draft_vocab_size is not None and target_vocab_size is not None:
            self.d2t_weight_ = ParameterWeight(
                weight_name="d2t",
                data_type=torch.int64,
                weight_shape=(draft_vocab_size,),
            )
            self.t2d_weight_ = ParameterWeight(
                weight_name="t2d",
                data_type=torch.bool,
                weight_shape=(target_vocab_size,),
            )

        self.lm_head_weight_: LMHeadWeight = LMHeadWeight(
            dim=hidden_size,
            vocab_size=vocab_size,
            weight_name="lm_head.weight",
            data_type=self.data_type_,
        )
        self.wte_weight_ = EmbeddingWeight(
            dim=hidden_size,
            vocab_size=vocab_size,
            weight_name="embed_tokens.weight",
            data_type=self.data_type_,
        )

        return
