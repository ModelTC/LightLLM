import torch
from lightllm.models.llama.layer_infer.post_layer_infer import LlamaPostLayerInfer


class Gemma4PostLayerInfer(LlamaPostLayerInfer):
    """
    Same final RMSNorm + tied lm_head path as Llama, with an extra tanh-based
    transform before sampling: logits = softcap * tanh(logits / softcap).
    """

    def __init__(self, network_config):
        super().__init__(network_config)
        self.final_logit_softcapping = float(network_config.get("final_logit_softcapping"))

    def _apply_logit_postprocessing(self, logits: torch.Tensor) -> torch.Tensor:
        if self.final_logit_softcapping is not None and self.final_logit_softcapping > 0:
            cap = self.final_logit_softcapping
            # The historical dense path first materializes FP32 logits and then
            # applies the softcap. Preserve that numerical contract when the
            # transform runs on a local BF16/FP16 vocabulary shard.
            return torch.tanh(logits.float() / cap) * cap
        return logits
