from lightllm.models.llama.layer_infer.post_layer_infer import LlamaPostLayerInfer
from .hyper_connection import hc_head
from ..infer_struct import DeepseekV4InferStateInfo


class DeepseekV4PostLayerInfer(LlamaPostLayerInfer):
    """Collapse the hc_mult residual streams (hc_head) to [T, hidden], then final norm + lm_head."""

    def token_forward(self, input_embdings, infer_state: DeepseekV4InferStateInfo, layer_weight):
        cfg = layer_weight.network_config_
        collapsed = hc_head(
            input_embdings,
            layer_weight.hc_head_fn_.weight,
            layer_weight.hc_head_scale_.weight,
            layer_weight.hc_head_base_.weight,
            cfg["hc_mult"],
            cfg["hidden_size"],
            cfg.get("hc_eps", 1e-6),
        )
        return super().token_forward(collapsed, infer_state, layer_weight)
