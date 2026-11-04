from lightllm.models.qwen3_dflash.adapter import Qwen3DFlashModelAdapter
from lightllm.models.qwen3_dspark.layer_infer.post_layer_infer import Qwen3DSparkPostLayerInfer
from lightllm.models.qwen3_dspark.layer_weights.pre_and_post_layer_weight import (
    Qwen3DSparkPreAndPostLayerWeight,
    Qwen3DSparkSharedPreAndPostLayerWeight,
)


class Qwen3DSparkModelAdapter(Qwen3DFlashModelAdapter):
    pre_and_post_weight_class = Qwen3DSparkPreAndPostLayerWeight
    post_layer_infer_class = Qwen3DSparkPostLayerInfer

    def build_recipe(self):
        pre, post, layers = super().build_recipe()
        post.heads_use_final_norm_ = "DSparkDraftModel" in self.config.get("architectures", ())
        return pre, post, layers

    def declare_weight_layout(self, names):
        vocab_weights = {"embed_tokens.weight", "lm_head.weight"}
        present = vocab_weights.intersection(names)
        if present == vocab_weights:
            return
        if present or "DSparkDraftModel" not in self.config.get("architectures", ()):
            raise ValueError("DSpark requires both vocabulary weights, or the shared DSparkDraftModel layout")
        for name in ("wte_weight_", "lm_head_weight_"):
            weight = getattr(self.main_weights, name)
            if weight.dim != self.config["hidden_size"] or weight.vocab_size != self.config["vocab_size"]:
                raise ValueError("shared DSpark vocabulary weights must match the target hidden size and vocabulary")
            if weight.data_type_ != self.dtype or not weight.verify_load():
                raise ValueError("shared DSpark vocabulary weights must be loaded in the draft dtype")
        self.pre_and_post_weight_class = Qwen3DSparkSharedPreAndPostLayerWeight
        self.share_main_weights = True
