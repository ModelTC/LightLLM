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
from lightllm.models.deepseek_v4.triton_kernel.quant_convert import dequant_fp8_block_to_bf16


class DeepseekV4MTPPreAndPostLayerWeight(PreAndPostLayerWeight):
    def __init__(self, data_type, network_config, quant_cfg: Quantcfg):
        super().__init__(data_type, network_config)
        self.quant_cfg: Quantcfg = quant_cfg

        hidden = network_config["hidden_size"]
        vocab = network_config["vocab_size"]
        hc_mult = network_config["hc_mult"]
        layer_idx = network_config["n_layer"]
        prefix = "mtp.0"

        self.wte_weight_ = EmbeddingWeight(
            dim=hidden,
            vocab_size=vocab,
            weight_name=f"{prefix}.emb.tok_emb.weight",
            data_type=self.data_type_,
        )
        self.lm_head_weight_ = LMHeadWeight(
            dim=hidden,
            vocab_size=vocab,
            weight_name=f"{prefix}.head.weight",
            data_type=self.data_type_,
        )
        self.final_norm_weight_ = RMSNormWeight(
            dim=hidden,
            weight_name=f"{prefix}.norm.weight",
            data_type=self.data_type_,
        )

        self.e_proj_weight_ = ROWMMWeight(
            in_dim=hidden,
            out_dims=[hidden],
            weight_names=f"{prefix}.e_proj.weight",
            data_type=self.data_type_,
            quant_method=self.quant_cfg.get_quant_method(layer_idx, "e_proj"),
            tp_rank=0,
            tp_world_size=1,
        )
        self.h_proj_weight_ = ROWMMWeight(
            in_dim=hidden,
            out_dims=[hidden],
            weight_names=f"{prefix}.h_proj.weight",
            data_type=self.data_type_,
            quant_method=self.quant_cfg.get_quant_method(layer_idx, "h_proj"),
            tp_rank=0,
            tp_world_size=1,
        )
        self.enorm_weight_ = RMSNormWeight(
            dim=hidden,
            weight_name=f"{prefix}.enorm.weight",
            data_type=self.data_type_,
        )
        self.hnorm_weight_ = RMSNormWeight(
            dim=hidden,
            weight_name=f"{prefix}.hnorm.weight",
            data_type=self.data_type_,
        )

        self.hc_head_fn_ = ParameterWeight(
            weight_name=f"{prefix}.hc_head_fn",
            data_type=torch.float32,
            weight_shape=(hc_mult, hc_mult * hidden),
        )
        self.hc_head_base_ = ParameterWeight(
            weight_name=f"{prefix}.hc_head_base",
            data_type=torch.float32,
            weight_shape=(hc_mult,),
        )
        self.hc_head_scale_ = ParameterWeight(
            weight_name=f"{prefix}.hc_head_scale",
            data_type=torch.float32,
            weight_shape=(1,),
        )
        return

    def load_hf_weights(self, weights):
        self._dequant_in_place(weights)
        return super().load_hf_weights(weights)

    def _dequant_in_place(self, weights):
        for attr in (self.e_proj_weight_, self.h_proj_weight_):
            for weight_name, scale_name in zip(attr.weight_names, attr.weight_scale_names):
                scale_key = weight_name[: -len(".weight")] + ".scale"
                if scale_key not in weights:
                    continue
                if scale_name is None:
                    weights[weight_name] = dequant_fp8_block_to_bf16(weights[weight_name], weights[scale_key]).to(
                        self.data_type_
                    )
                else:
                    weights[scale_name] = weights[scale_key].to(torch.float32)
                del weights[scale_key]
        return
