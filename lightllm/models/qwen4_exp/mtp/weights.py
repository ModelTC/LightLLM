from dataclasses import dataclass
from functools import partial

import torch.nn.functional as F

from lightllm.common.basemodel import PreAndPostLayerWeight
from lightllm.common.gated_residual.kernels import grouped_norm
from ..layer_weights.components import ComponentWeight, gated_residual_weight
from ..layer_weights.transformer_layer_weight import Qwen4ExpTransformerLayerWeight


@dataclass(frozen=True)
class MTPFusion:
    parameters: dict
    branches: int
    eps: float

    def __call__(self, embedding, hidden):
        if embedding.ndim != 2 or not embedding.is_contiguous() or not hidden.is_contiguous():
            raise ValueError("Qwen MTP fusion requires contiguous embedding and hidden matrices")
        width = embedding.shape[-1]
        if hidden.shape != (embedding.shape[0], self.branches * width):
            raise ValueError("Qwen MTP requires the full pre-mixer HC hidden stream")
        weights = self.parameters
        embedding = grouped_norm(embedding, weights["mtp.pre_fc_norm_embedding.weight"], 1, self.eps)
        hidden = grouped_norm(hidden, weights["mtp.pre_fc_norm_hidden.weight"], 1, self.eps)
        embedding = F.linear(embedding, weights["mtp.fc_embedding.weight"])
        hidden = F.linear(hidden.view(-1, self.branches, width), weights["mtp.fc_hidden.weight"])
        return (hidden + embedding[:, None]).flatten(1)


class Qwen4ExpMTPPreAndPostLayerWeight(PreAndPostLayerWeight):
    def __init__(self, data_type, network_config):
        super().__init__(data_type, network_config)
        hidden, branches = network_config["hidden_size"], network_config["hc_count"]
        specs = {
            "mtp.pre_fc_norm_embedding.weight": ((hidden,), data_type),
            "mtp.pre_fc_norm_hidden.weight": ((hidden * branches,), data_type),
            "mtp.fc_embedding.weight": ((hidden, hidden), data_type),
            "mtp.fc_hidden.weight": ((hidden, hidden), data_type),
        }
        self.fusion = ComponentWeight(specs, partial(MTPFusion, branches=branches, eps=network_config["rms_norm_eps"]))
        self.hyper_connection_mixer = gated_residual_weight(
            "mtp.hyper_connection_mixer", network_config, data_type, use_combine=False
        )
        self.wte_weight_ = self.lm_head_weight_ = None


class Qwen4ExpMTPTransformerLayerWeight(Qwen4ExpTransformerLayerWeight):
    def load_hf_weights(self, weights):
        prefix = "mtp.layers.0."
        mapped = {
            "model.layers.0." + name[len(prefix) :]: value for name, value in weights.items() if name.startswith(prefix)
        }
        super().load_hf_weights(mapped)
