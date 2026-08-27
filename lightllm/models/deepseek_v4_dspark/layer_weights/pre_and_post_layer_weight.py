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
from lightllm.models.deepseek_v4.triton_kernel.quant_convert import (
    dequant_fp8_block_to_bf16,
)


class DeepseekV4DSparkPreAndPostLayerWeight(PreAndPostLayerWeight):
    """Shared vocabulary weights plus the first/last DSpark stage heads."""

    def __init__(self, data_type, network_config, quant_cfg: Quantcfg):
        super().__init__(data_type, network_config)
        self.quant_cfg = quant_cfg

        hidden = network_config["hidden_size"]
        vocab = network_config["vocab_size"]
        hc_mult = network_config["hc_mult"]
        markov_rank = network_config["dspark_markov_rank"]
        target_hidden_size = hidden * len(network_config["dspark_target_layer_ids"])
        first_layer_idx = network_config["n_layer"]
        last_stage = network_config["dspark_layer_num"] - 1
        last_prefix = f"mtp.{last_stage}"

        # Replaced with the already-loaded target weights by the model.
        self.wte_weight_ = EmbeddingWeight(
            dim=hidden,
            vocab_size=vocab,
            weight_name="embed.weight",
            data_type=self.data_type_,
        )
        self.lm_head_weight_ = LMHeadWeight(
            dim=hidden,
            vocab_size=vocab,
            weight_name="head.weight",
            data_type=self.data_type_,
        )

        self.main_proj_weight_ = ROWMMWeight(
            in_dim=target_hidden_size,
            out_dims=[hidden],
            weight_names="mtp.0.main_proj.weight",
            data_type=self.data_type_,
            quant_method=self.quant_cfg.get_quant_method(first_layer_idx, "main_proj"),
            tp_rank=0,
            tp_world_size=1,
        )
        self.main_norm_weight_ = RMSNormWeight(
            dim=hidden,
            weight_name="mtp.0.main_norm.weight",
            data_type=self.data_type_,
        )

        self.final_norm_weight_ = RMSNormWeight(
            dim=hidden,
            weight_name=f"{last_prefix}.norm.weight",
            data_type=self.data_type_,
        )
        self.hc_head_fn_ = ParameterWeight(
            weight_name=f"{last_prefix}.hc_head_fn",
            data_type=torch.float32,
            weight_shape=(hc_mult, hc_mult * hidden),
        )
        self.hc_head_base_ = ParameterWeight(
            weight_name=f"{last_prefix}.hc_head_base",
            data_type=torch.float32,
            weight_shape=(hc_mult,),
        )
        self.hc_head_scale_ = ParameterWeight(
            weight_name=f"{last_prefix}.hc_head_scale",
            data_type=torch.float32,
            weight_shape=(1,),
        )

        # W1 is replicated to avoid one TP collective in every sequential Markov step.
        self.markov_w1_weight_ = EmbeddingWeight(
            dim=markov_rank,
            vocab_size=vocab,
            weight_name=f"{last_prefix}.markov_head.markov_w1.weight",
            data_type=self.data_type_,
            tp_rank=0,
            tp_world_size=1,
        )
        self.markov_w2_weight_ = LMHeadWeight(
            dim=markov_rank,
            vocab_size=vocab,
            weight_name=f"{last_prefix}.markov_head.markov_w2.weight",
            data_type=self.data_type_,
        )
        self.confidence_head_weight_ = ROWMMWeight(
            in_dim=hidden + markov_rank,
            out_dims=[1],
            weight_names=f"{last_prefix}.confidence_head.proj.weight",
            bias_names=None,
            data_type=torch.float32,
            quant_method=None,
            tp_rank=0,
            tp_world_size=1,
        )

        # Reused by the generic DSpark Markov implementation.
        self.markov_rank = markov_rank
        self.markov_head_type = "vanilla"

    def load_hf_weights(self, weights):
        self._dequant_main_proj_in_place(weights)
        return super().load_hf_weights(weights)

    def _dequant_main_proj_in_place(self, weights):
        weight_name = self.main_proj_weight_.weight_names[0]
        scale_key = weight_name[: -len(".weight")] + ".scale"
        if scale_key not in weights:
            return

        scale_name = self.main_proj_weight_.weight_scale_names[0]
        if scale_name is None:
            weights[weight_name] = dequant_fp8_block_to_bf16(
                weights[weight_name], weights[scale_key]
            ).to(self.data_type_)
        else:
            weights[scale_name] = weights[scale_key].to(torch.float32)
        del weights[scale_key]
