import torch
from lightllm.models.registry import ModelRegistry
from lightllm.models.deepseek2.model import Deepseek2TpPartModel
from lightllm.models.glm4_moe_lite.layer_infer.transformer_layer_infer import Glm4MoeLiteTransformerLayerInfer
from lightllm.models.glm4_moe_lite.layer_weights.transformer_layer_weight import Glm4MoeLiteTransformerLayerWeight
from lightllm.models.glm4_moe_lite.infer_struct import Glm4MoeLiteInferStateInfo
from lightllm.distributed.communication_op import dist_group_manager
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


@ModelRegistry("glm4_moe_lite")
class Glm4MoeLiteTpPartModel(Deepseek2TpPartModel):

    transformer_weight_class = Glm4MoeLiteTransformerLayerWeight
    transformer_layer_infer_class = Glm4MoeLiteTransformerLayerInfer
    infer_state_class = Glm4MoeLiteInferStateInfo

    def __init__(self, kvargs):
        super().__init__(kvargs)

    def _init_config(self):
        super()._init_config()

        if "moe_layer_freq" not in self.config and self.config.get("n_routed_experts"):
            self.config["moe_layer_freq"] = 1

        if "routed_scaling_factor" not in self.config:
            self.config["routed_scaling_factor"] = 1.8

        if "topk_method" not in self.config:
            self.config["topk_method"] = "noaux_tc"

        if "scoring_func" not in self.config:
            self.config["scoring_func"] = "sigmoid"

        logger.info(
            f"GLM-4.7-Flash config: "
            f"n_routed_experts={self.config.get('n_routed_experts')}, "
            f"n_shared_experts={self.config.get('n_shared_experts')}, "
            f"num_experts_per_tok={self.config.get('num_experts_per_tok')}, "
            f"first_k_dense_replace={self.config.get('first_k_dense_replace')}, "
            f"routed_scaling_factor={self.config.get('routed_scaling_factor')}, "
            f"scoring_func={self.config.get('scoring_func')}"
        )

    def _init_custom(self):
        self._init_to_get_yarn_rotary()
        dist_group_manager.new_deepep_group(self.config["n_routed_experts"], self.config["hidden_size"])

    def _init_to_get_yarn_rotary(self):
        rope_scaling = self.config.get("rope_scaling")

        if rope_scaling is None:
            self._init_glm4_standard_rotary()
        else:
            super()._init_to_get_yarn_rotary()

    def _init_glm4_standard_rotary(self):
        rope_theta = self.config.get("rope_theta", 1000000.0)
        qk_rope_head_dim = self.config.get("qk_rope_head_dim", 64)
        max_position_embeddings = self.config.get("max_position_embeddings", 202752)

        dim = qk_rope_head_dim

        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, device="cpu", dtype=torch.float32) / dim))

        max_seq_len = max(max_position_embeddings, self.max_seq_length)
        t = torch.arange(max_seq_len, device="cpu", dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)

        self._cos_cached = torch.cos(freqs).to(self.data_type).cuda()
        self._sin_cached = torch.sin(freqs).to(self.data_type).cuda()
