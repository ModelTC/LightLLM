import os
from lightllm.utils.log_utils import init_logger
from lightllm.utils.envs_utils import enable_env_vars
from lightllm.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import ROWMMWeight, FusedMoeWeight
from lightllm.utils.envs_utils import get_env_start_args

logger = init_logger(__name__)


class MixtralTransformerLayerWeight(LlamaTransformerLayerWeight):
    def __init__(self, layer_num, data_type, network_config, quant_cfg=None):
        super().__init__(
            layer_num,
            data_type,
            network_config,
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

        self.moe_gate = ROWMMWeight(
            in_dim=self.n_embed,
            out_dims=[self.n_routed_experts],
            weight_names=self.moe_gate_weight_name,
            data_type=self.data_type_,
            bias_names=self.moe_gate_bias_name,
            quant_method=self.get_quant_method("moe_gate"),
            tp_rank=0,
            tp_world_size=1,  # no tensor parallelism
        )
        assert get_env_start_args().enable_ep_moe, "Mixtral only support tp mode."
        self.experts = FusedMoeWeight(
            gate_proj_name="w1",
            down_proj_name="w2",
            up_proj_name="w3",
            e_score_correction_bias_name="",
            weight_prefix=f"model.layers.{self.layer_num_}.block_sparse_moe.experts",
            n_routed_experts=self.n_routed_experts,
            hidden_size=self.n_embed,
            moe_intermediate_size=inter_size,
            data_type=self.data_type_,
            quant_method=self.quant_cfg.get_quant_method(self.layer_num_, "fused_moe"),
            layer_num=self.layer_num_,
            network_config=self.network_config_,
        )
