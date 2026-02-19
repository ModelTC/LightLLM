import torch
from lightllm.models.llama.layer_infer.post_layer_infer import LlamaPostLayerInfer
from lightllm.models.qwen3next_mtp.layer_weights.pre_and_post_layer_weight import Qwen3NextMTPPreAndPostLayerWeight
from lightllm.models.qwen3next.triton_kernel.gemma_rmsnorm import gemma_rmsnorm_forward


class Qwen3NextMTPPostLayerInfer(LlamaPostLayerInfer):
    """
    Qwen3Next MTP Post Layer Inference.
    Uses gemma_rmsnorm for normalization (same as Qwen3Next).
    """

    def _norm(self, input, infer_state, layer_weight: Qwen3NextMTPPreAndPostLayerWeight) -> torch.Tensor:
        out = self.alloc_tensor(input.shape, input.dtype)
        gemma_rmsnorm_forward(input, layer_weight.final_norm_weight_.weight, self.eps_, out=out)
        return out
