from lightllm.models.qwen3_5.layer_weights.transformer_layer_weight import (
    Qwen35TransformerLayerWeight,
)


def rename_mtp_weight_keys(weights):
    for name in list(weights.keys()):
        if name.startswith("model."):
            weights.pop(name)

    for name in list(weights.keys()):
        if name.startswith("mtp."):
            weights[f"model.{name[len('mtp.'):]}"] = weights.pop(name)


class Qwen3_5MTPTransformerLayerWeight(Qwen35TransformerLayerWeight):
    def load_hf_weights(self, weights):
        rename_mtp_weight_keys(weights)
        return super().load_hf_weights(weights)
