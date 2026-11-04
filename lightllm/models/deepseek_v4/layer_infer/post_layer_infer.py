from lightllm.common.basemodel.triton_kernel.mhc import hc_head, hc_post
from lightllm.models.llama.layer_infer.post_layer_infer import LlamaPostLayerInfer
from ..infer_struct import DeepseekV4InferStateInfo


class DeepseekV4PostLayerInfer(LlamaPostLayerInfer):
    """Collapse the hc_mult residual streams (hc_head) to [T, hidden], then final norm + lm_head."""

    def token_forward(self, input_embdings, infer_state: DeepseekV4InferStateInfo, layer_weight):
        cfg = layer_weight.network_config_
        if isinstance(input_embdings, tuple):
            # Autotune can stop before the last layer. Finish its pending post.
            input_embdings = hc_post(*input_embdings, streams=cfg["hc_mult"])
        out = self.alloc_tensor(
            (input_embdings.shape[0], cfg["hidden_size"]), dtype=input_embdings.dtype, device=input_embdings.device
        )
        collapsed = hc_head(
            x=input_embdings,
            fn=layer_weight.hc_head_fn_.weight,
            scale=layer_weight.hc_head_scale_.weight,
            base=layer_weight.hc_head_base_.weight,
            streams=cfg["hc_mult"],
            rms_eps=cfg["rms_norm_eps"],
            hc_eps=cfg.get("hc_eps", 1e-6),
            out=out,
        )
        return super().token_forward(collapsed, infer_state, layer_weight)
