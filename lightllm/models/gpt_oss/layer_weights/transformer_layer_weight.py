import os
import torch
import numpy as np

from lightllm.common.basemodel.layer_weights.meta_weights.gpt_oss_fused_moe_weight_tp import GPTOSSFusedMoeWeightTP
from lightllm.common.basemodel.layer_weights.meta_weights.mm_weight.rowmm_weight import ROWMMWeight
from lightllm.common.basemodel.layer_weights.meta_weights.norm_weight import NormWeight, TpNormWeight
from lightllm.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class GptOssTransformerLayerWeight(LlamaTransformerLayerWeight):
    def __init__(
        self,
        layer_num,
        data_type,
        network_config,
        mode=[],
        quant_cfg=None,
    ):
        super().__init__(layer_num, data_type, network_config, mode, quant_cfg)
        return

    def _init_moe(self):
        moe_mode = os.getenv("MOE_MODE", "TP")
        moe_intermediate_size = self.network_config_["intermediate_size"]
        n_routed_experts = self.network_config_["num_local_experts"]
        assert moe_mode in ["TP"], "For now, GPT-OSS type model only support MOE TP mode."

        self.moe_gate = ROWMMWeight(
            weight_name=self._router_weight_name,
            data_type=self.data_type_,
            layer_num=self.layer_num_,
            bias_name=self._router_bias_name,
            name="moe_gate",
            tp_rank=0,
            tp_world_size=1,
        )

        self.experts = GPTOSSFusedMoeWeightTP(
            gate_up_proj_name="gate_up_proj",  # diff with FusedMoeWeightTP
            down_proj_name="down_proj",
            e_score_correction_bias_name="",
            weight_prefix=f"model.layers.{self.layer_num_}.mlp.experts",
            n_routed_experts=n_routed_experts,
            split_inter_size=moe_intermediate_size // self.tp_world_size_,
            data_type=self.data_type_,
            network_config=self.network_config_,
            layer_num=self.layer_num_,
            world_size=self.tp_world_size_,  # diff with FusedMoeWeightTP
            quant_cfg=self.quant_cfg,
            num_fused_shared_experts=0,
        )

    def _init_weight_names(self):
        super()._init_weight_names()

        self._attn_sink_name = f"model.layers.{self.layer_num_}.self_attn.sinks"

        self._q_bias_name = f"model.layers.{self.layer_num_}.self_attn.q_proj.bias"
        self._k_bias_name = f"model.layers.{self.layer_num_}.self_attn.k_proj.bias"
        self._v_bias_name = f"model.layers.{self.layer_num_}.self_attn.v_proj.bias"
        self._o_bias_name = f"model.layers.{self.layer_num_}.self_attn.o_proj.bias"

        self._router_bias_name = f"model.layers.{self.layer_num_}.mlp.router.bias"
        self._router_weight_name = f"model.layers.{self.layer_num_}.mlp.router.weight"

    def _init_weight(self):
        super()._init_weight()

        n_split_head = self.network_config_["num_attention_heads"] // self.tp_world_size_
        self.attn_sinks = TpNormWeight(self._attn_sink_name, torch.bfloat16, n_split_head)

    def _init_ffn(self):
        self._init_moe()
