import os
from lightllm.utils.log_utils import init_logger
from lightllm.utils.envs_utils import enable_env_vars
from lightllm.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import (
    ROWMMWeight,
    FusedMoeWeightTP,
    FusedMoeWeightEP,
)

logger = init_logger(__name__)


class MixtralTransformerLayerWeight(LlamaTransformerLayerWeight):
    def __init__(self, layer_num, data_type, network_config, mode=[], quant_cfg=None):
        super().__init__(
            layer_num,
            data_type,
            network_config,
            mode,
            quant_cfg=quant_cfg,
        )
        return

    def _parse_config(self):
        super()._parse_config()
        self.n_routed_experts = self.network_config_["num_local_experts"]

    def _init_weight_names(self):
        super()._init_weight_names()
        self.moe_gate_weight_name = f"model.layers.{self.layer_num_}.block_sparse_moe.gate.weight"
        self.moe_gate_bias_name = None

    def _init_ffn(self):
        self._init_moe()

    def _init_moe(self):
        inter_size = self.network_config_["intermediate_size"]
        split_inter_size = inter_size // self.tp_world_size_

        self.moe_gate = ROWMMWeight(
            weight_name=self.moe_gate_weight_name,
            data_type=self.data_type_,
            bias_name=self.moe_gate_bias_name,
            quant_cfg=self.quant_cfg,
            layer_num=self.layer_num_,
            name="moe_gate",
            tp_rank=0,
            tp_world_size=1,  # no tensor parallelism
        )

        moe_mode = os.getenv("MOE_MODE", "TP")
        assert moe_mode in ["TP"], f"Unsupported moe mode: {moe_mode}"

        if moe_mode == "TP":
            self.experts = FusedMoeWeightTP(
                gate_proj_name="w1",
                down_proj_name="w2",
                up_proj_name="w3",
                e_score_correction_bias_name="",
                weight_prefix=f"model.layers.{self.layer_num_}.block_sparse_moe.experts",
                n_routed_experts=self.n_routed_experts,
                split_inter_size=split_inter_size,
                data_type=self.data_type_,
                network_config=self.network_config_,
                layer_num=self.layer_num_,
                quant_cfg=self.quant_cfg,
                num_fused_shared_experts=0,
            )
        else:
            raise ValueError(f"Unsupported moe mode: {moe_mode}")
