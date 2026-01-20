import torch
from lightllm.models.registry import ModelRegistry
from lightllm.models.llama.model import LlamaTpPartModel
from lightllm.models.glm4_moe.layer_weights.transformer_layer_weight import Glm4MoeTransformerLayerWeight
from lightllm.models.glm4_moe.layer_infer.transformer_layer_infer import Glm4MoeTransformerLayerInfer
from lightllm.distributed.communication_op import dist_group_manager


@ModelRegistry("glm4_moe")
class Glm4MoeTpPartModel(LlamaTpPartModel):
    # weight class
    transformer_weight_class = Glm4MoeTransformerLayerWeight

    # infer class
    transformer_layer_infer_class = Glm4MoeTransformerLayerInfer

    def __init__(self, kvargs):
        super().__init__(kvargs)
        return

    def _init_config(self):
        super()._init_config()
        if "num_key_value_heads" not in self.config:
            self.config["num_key_value_heads"] = 8
        self.config["scoring_func"] = "sigmoid"
        return

    def _init_some_value(self):
        super()._init_some_value()
        # GLM-4.7 uses partial rotary with factor 0.5
        self.partial_rotary_factor = self.config.get("partial_rotary_factor", 0.5)
        # Store head_dim explicitly
        self.head_dim_ = self.config.get("head_dim", self.config["hidden_size"] // self.config["num_attention_heads"])
        return

    def _init_custom(self):
        # Initialize partial rotary embeddings
        self._init_to_get_partial_rotary()
        # Initialize expert parallelism group for 160 routed experts
        n_routed_experts = self.config.get("n_routed_experts", 160)
        dist_group_manager.new_deepep_group(n_routed_experts, self.config["hidden_size"])
        return

    def _init_to_get_partial_rotary(self, default_base=10000):
        partial_rotary_factor = self.config.get("partial_rotary_factor", 0.5)
        partial_head_dim = int(partial_rotary_factor * self.head_dim_)

        if self.config.get("rope_scaling", {}) is None:
            rope_scaling_factor = 1.0
        else:
            rope_scaling_factor = self.config.get("rope_scaling", {}).get("factor", 1.0)

        base = self.config.get("rope_theta", float(default_base))

        max_position_embeddings = self.config.get("max_position_embeddings", 202752)
        max_seq_len = max(max_position_embeddings * rope_scaling_factor, self.max_seq_length)

        inv_freq = 1.0 / (
            base ** (torch.arange(0, partial_head_dim, 2, device="cpu", dtype=torch.float32) / partial_head_dim)
        )
        t = torch.arange(int(max_seq_len + 1024 * 128), device="cpu", dtype=torch.float32) / rope_scaling_factor
        freqs = torch.outer(t, inv_freq)

        self._cos_cached = torch.cos(freqs).to(self.data_type).cuda()
        self._sin_cached = torch.sin(freqs).to(self.data_type).cuda()
        return

    def _verify_params(self):
        assert self.load_way in ["HF", "DS"], "GLM-4.7 MoE only supports HF and DS format to load"
        assert self.config["num_key_value_heads"] % self.tp_world_size_ == 0
        assert self.config["num_attention_heads"] % self.tp_world_size_ == 0
        return

    def autotune_layers(self):
        # For GLM-4.7, first 3 layers are dense, rest are MoE
        # Autotune first few dense layers + at least one MoE layer
        return self.config.get("first_k_dense_replace", 3) + 1
