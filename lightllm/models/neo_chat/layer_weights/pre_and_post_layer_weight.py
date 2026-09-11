from lightllm.models.qwen2.layer_weights.pre_and_post_layer_weight import (
    Qwen2PreAndPostLayerWeight,
)


def strip_language_model_prefix(weights) -> None:
    """Expose Neo's nested language-model weights to ordinary Qwen loaders."""

    prefix = "language_model."
    for name in list(weights):
        if name.startswith(prefix):
            weights[name.removeprefix(prefix)] = weights.pop(name)


class NeoChatPreAndPostLayerWeight(Qwen2PreAndPostLayerWeight):
    def load_hf_weights(self, weights):
        # This loader runs before transformer-layer loaders on each shard, so
        # the one in-place rename also makes every layer key visible.
        strip_language_model_prefix(weights)
        super().load_hf_weights(weights)
