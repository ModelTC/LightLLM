import torch

from lightllm.distributed.communication_op import dist_group_manager
from lightllm.models.deepseek3_2.model import Deepseek3_2TpPartModel
from lightllm.models.glm5_2.layer_infer.transformer_layer_infer import Glm5_2TransformerLayerInfer
from lightllm.models.glm5_2.layer_weights.transformer_layer_weight import Glm5_2TransformerLayerWeight
from lightllm.models.registry import ModelRegistry


@ModelRegistry("glm_moe_dsa")
class Glm5_2TpPartModel(Deepseek3_2TpPartModel):
    transformer_weight_class = Glm5_2TransformerLayerWeight
    transformer_layer_infer_class = Glm5_2TransformerLayerInfer

    def _init_config(self):
        super()._init_config()
        if "scoring_func" not in self.config:
            self.config["scoring_func"] = "sigmoid"
        if self.config.get("rope_theta") is None:
            self.config["rope_theta"] = self.config.get("rope_parameters", {}).get("rope_theta", 1000000.0)

    def _init_custom(self):
        self._init_glm5_2_rotary()
        dist_group_manager.new_deepep_group(
            self.config["n_routed_experts"],
            self.config["hidden_size"],
            self.config.get("num_experts_per_tok", 1),
            self.config.get("moe_intermediate_size", self.config.get("intermediate_size")),
        )

    def _create_inferstate(self, model_input, microbatch_index: int = 0):
        infer_state = super()._create_inferstate(model_input, microbatch_index=microbatch_index)
        infer_state.glm5_2_model_input = model_input
        infer_state.glm5_2_reuse_mtp_topk_indices = (
            getattr(self, "is_mtp_draft_model", False)
            and self.config.get("index_share_for_mtp_iteration", False)
            and not model_input.is_prefill
        )
        return infer_state

    def _init_glm5_2_rotary(self):
        rope_theta = self.config.get("rope_theta", 8000000.0)
        qk_rope_head_dim = self.config.get("qk_rope_head_dim", 64)
        max_position_embeddings = self.config.get("max_position_embeddings", 1048576)

        inv_freq = 1.0 / (
            rope_theta ** (torch.arange(0, qk_rope_head_dim, 2, device="cpu", dtype=torch.float32) / qk_rope_head_dim)
        )
        max_seq_len = max(max_position_embeddings, self.max_seq_length)
        t = torch.arange(max_seq_len, device="cpu", dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)

        self._cos_cached = torch.cos(freqs).to(self.data_type).cuda()
        self._sin_cached = torch.sin(freqs).to(self.data_type).cuda()
