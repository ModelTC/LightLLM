import os
import torch
import numpy as np

from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.gpt_oss_fused_moe_weight_tp import (
    GPTOSSFusedMoeWeightTP,
)
from lightllm.common.basemodel.layer_weights.meta_weights.mm_weight import ROWMMWeight
from lightllm.common.basemodel.layer_weights.meta_weights import TpAttSinkWeight
from lightllm.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight
from lightllm.utils.log_utils import init_logger
from lightllm.utils.envs_utils import get_env_start_args

logger = init_logger(__name__)


class GptOssTransformerLayerWeight(LlamaTransformerLayerWeight):
    def __init__(
        self,
        layer_num,
        data_type,
        network_config,
        quant_cfg=None,
    ):
        super().__init__(layer_num, data_type, network_config, quant_cfg)
        return

    def _init_moe(self):
        enable_ep_moe = get_env_start_args().enable_ep_moe
        moe_intermediate_size = self.network_config_["intermediate_size"]
        n_routed_experts = self.network_config_["num_local_experts"]
        assert not enable_ep_moe, "For now, GPT-OSS type model only support MOE TP mode."

        self.moe_gate = ROWMMWeight(
            in_dim=self.n_embed,
            out_dims=[n_routed_experts],
            weight_names=self._router_weight_name,
            data_type=self.data_type_,
            bias_names=self._router_bias_name,
            quant_method=self.get_quant_method("moe_gate"),
            tp_rank=0,
            tp_world_size=1,
        )

        self.experts = GPTOSSFusedMoeWeightTP(
            gate_up_proj_name="gate_up_proj",  # diff with FusedMoeWeightTP
            down_proj_name="down_proj",
            e_score_correction_bias_name="",
            weight_prefix=f"model.layers.{self.layer_num_}.mlp.experts",
            n_routed_experts=n_routed_experts,
            hidden_size=self.n_embed,
            moe_intermediate_size=moe_intermediate_size,
            data_type=self.data_type_,
            quant_method=self.quant_cfg.get_quant_method(self.layer_num_, "fused_moe"),
            num_fused_shared_experts=0,
            layer_num=self.layer_num_,
            network_config=self.network_config_,
            moe_layer_index=self.layer_num_,
        )

    def _init_weight_names(self):
        super()._init_weight_names()

        self._q_bias_name = f"model.layers.{self.layer_num_}.self_attn.q_proj.bias"
        self._k_bias_name = f"model.layers.{self.layer_num_}.self_attn.k_proj.bias"
        self._v_bias_name = f"model.layers.{self.layer_num_}.self_attn.v_proj.bias"
        self._o_bias_name = f"model.layers.{self.layer_num_}.self_attn.o_proj.bias"

        self._router_bias_name = f"model.layers.{self.layer_num_}.mlp.router.bias"
        self._router_weight_name = f"model.layers.{self.layer_num_}.mlp.router.weight"

    def _init_weight(self):
        super()._init_weight()

        self.attn_sinks = TpAttSinkWeight(
            all_q_head_num=self.q_head_num_,
            weight_name=f"model.layers.{self.layer_num_}.self_attn.sinks",
            data_type=torch.bfloat16,
        )

    def _init_ffn(self):
        self._init_moe()
