from lightllm.models.qwen3_dspark.adapter import Qwen3DSparkModelAdapter
from lightllm.models.qwen3_dflash.model import Qwen3DFlashModel
from lightllm.models.qwen3_dflash.adapter import Qwen3DFlashModelAdapter
from lightllm.models.llama.model import LlamaTpPartModel
from lightllm.models.qwen3_dspark.layer_infer.post_layer_infer import Qwen3DSparkPostLayerInfer
from lightllm.models.qwen3_dspark.layer_weights.pre_and_post_layer_weight import (
    Qwen3DSparkPreAndPostLayerWeight,
    Qwen3DSparkSharedPreAndPostLayerWeight,
)


class Qwen3DSparkModel(Qwen3DFlashModel):
    """Qwen3 DSpark draft model.

    DSpark keeps the DFlash block backbone.  Its extra Markov/confidence heads
    run in the post layer so the proposer can consume corrected logits through
    the same token-generation path as DFlash.
    """

    pre_and_post_weight_class = Qwen3DSparkPreAndPostLayerWeight
    post_layer_infer_class = Qwen3DSparkPostLayerInfer
    model_adapter_class = Qwen3DSparkModelAdapter

    @classmethod
    def normalize_model_config(cls, model_config: dict[str, object]) -> dict[str, object]:
        config = model_config.get("text_config", model_config).copy()
        draft_config = config.get("dflash_config", {})
        if not isinstance(draft_config, dict):
            raise ValueError("dflash_config must be an object in the DSpark checkpoint config")
        config.update(draft_config)
        config = super().normalize_model_config(config)
        rope = config["rope_parameters"]
        if "rope_theta" in rope and "rope_theta" not in config:
            config["rope_theta"] = rope["rope_theta"]
        config["rope_scaling"] = rope
        config["partial_rotary_factor"] = rope.get("partial_rotary_factor", 1.0)
        return config

    def _create_model_adapter(self, adapter_class):
        from lightllm.common.basemodel.layer_weights.hf_load_utils import checkpoint_weight_names

        adapter = super()._create_model_adapter(adapter_class)
        adapter.declare_weight_layout(checkpoint_weight_names(self.weight_dir_, self.weight_dict))
        return adapter

    def _init_custom(self):
        LlamaTpPartModel._init_custom(self)
        self.block_size = self.config["block_size"]
        self.mask_token_id = self.config["mask_token_id"]

    def _verify_params(self):
        self.config.setdefault("block_size", self.args.mtp_step)
        super()._verify_params()
