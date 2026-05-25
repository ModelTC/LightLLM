import torch
from lightllm.models.llama.layer_infer.post_layer_infer import LlamaPostLayerInfer


class Gemma4PostLayerInfer(LlamaPostLayerInfer):
    """
    Same final RMSNorm + tied lm_head path as Llama, with an extra tanh-based
    logit softcap at the end: logits = softcap * tanh(logits / softcap).
    """

    def __init__(self, network_config):
        super().__init__(network_config)
        self.final_logit_softcapping = float(network_config.get("final_logit_softcapping"))

    def token_forward(self, input_embdings, infer_state, layer_weight):
        logits = super().token_forward(input_embdings, infer_state, layer_weight)
        if self.final_logit_softcapping is not None and self.final_logit_softcapping > 0:
            cap = self.final_logit_softcapping
            logits = torch.tanh(logits / cap) * cap
        return logits
