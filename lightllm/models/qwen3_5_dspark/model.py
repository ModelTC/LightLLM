from lightllm.models.llama.model import LlamaTpPartModel
from lightllm.models.qwen3_dspark.model import Qwen3DSparkModel


class Qwen3_5DSparkModel(Qwen3DSparkModel):
    """Adapter for the current Qwen3 DSpark checkpoint with a Qwen3.5 target."""

    def _init_config(self):
        super()._init_config()
        self.config.update(self.config.get("dflash_config", {}))

        rope_parameters = self.config["rope_parameters"]
        if "rope_theta" in rope_parameters and "rope_theta" not in self.config:
            self.config["rope_theta"] = rope_parameters["rope_theta"]

        # The draft is Qwen3-style and owns a 1D rotary cache. Released
        # checkpoints rotate the full head, while target-shaped custom
        # checkpoints can retain Qwen3.5's partial rotary layout.
        self.config["rope_scaling"] = rope_parameters
        self.config["partial_rotary_factor"] = rope_parameters.get("partial_rotary_factor", 1.0)

    def _init_custom(self):
        # Draft and target use different rotary shapes, so the draft owns its rotary cache.
        LlamaTpPartModel._init_custom(self)
        self.block_size = self.config["block_size"]
        self.mask_token_id = self.config["mask_token_id"]

    def _init_mem_manager(self):
        layer_start = self.main_model.mem_manager.linear_config.get_main_model_full_att_layer_num()
        layer_start += sum(len(model.layers_infer) for model in self.mtp_previous_draft_models)
        self.mem_manager = self.main_model.mem_manager.get_kv_layout_view(
            self.config["num_key_value_heads"],
            self.config["head_dim"],
            layer_start=layer_start,
            layer_num=self.config["n_layer"],
        )
