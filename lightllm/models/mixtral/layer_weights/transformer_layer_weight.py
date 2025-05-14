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

    def _init_ffn(self, weights):
        self._init_moe(weights)

    def _init_moe(self, weights):
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
            tp_size=1,  # no tensor parallelism
        )

        load_func = FusedMoeWeightEP if enable_env_vars("ETP_MODE_ENABLED") else FusedMoeWeightTP
        self.experts = load_func(
            gate_proj_name="w1",
            down_proj_name="w2",
            up_proj_name="w3",
            weight_prefix=f"model.layers.{self.layer_num_}.block_sparse_moe.experts",
            n_routed_experts=self.n_routed_experts,
            split_inter_size=split_inter_size,
            data_type=self.data_type_,
        )
