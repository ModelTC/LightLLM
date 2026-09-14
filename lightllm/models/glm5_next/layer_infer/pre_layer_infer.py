# SPDX-License-Identifier: Apache-2.0

from lightllm.common.basemodel.triton_kernel.mhc import hc_expand
from lightllm.models.llama.layer_infer.pre_layer_infer import LlamaPreLayerInfer


class Glm5NextPreLayerInfer(LlamaPreLayerInfer):
    """Initialize mHC residual streams after token embedding and TP reduction."""

    def __init__(self, network_config):
        super().__init__(network_config)
        self.mhc_streams = network_config.get("hc_mult", 4)

    def context_forward(self, input_ids, infer_state, layer_weight):
        input_embeddings = super().context_forward(input_ids, infer_state, layer_weight)
        return hc_expand(input_embeddings, self.mhc_streams)

    def token_forward(self, input_ids, infer_state, layer_weight):
        input_embeddings = super().token_forward(input_ids, infer_state, layer_weight)
        return hc_expand(input_embeddings, self.mhc_streams)
