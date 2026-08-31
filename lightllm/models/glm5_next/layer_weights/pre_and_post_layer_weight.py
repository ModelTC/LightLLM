# SPDX-License-Identifier: Apache-2.0

from lightllm.models.llama.layer_weights.pre_and_post_layer_weight import (
    LlamaPreAndPostLayerWeight,
)


def add_language_model_aliases(weights: dict) -> None:
    """Expose GLM's nested language-model keys under LightLLM names."""

    prefix = "model.language_model."
    for name in list(weights):
        if name.startswith(prefix):
            weights.setdefault("model." + name[len(prefix) :], weights[name])


class Glm5NextPreAndPostLayerWeight(LlamaPreAndPostLayerWeight):
    def load_hf_weights(self, weights):
        add_language_model_aliases(weights)
        return super().load_hf_weights(weights)
