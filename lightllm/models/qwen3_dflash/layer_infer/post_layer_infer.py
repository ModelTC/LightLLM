import torch

from lightllm.models.llama.layer_infer.post_layer_infer import LlamaPostLayerInfer


class Qwen3DFlashPostLayerInfer(LlamaPostLayerInfer):
    def token_forward(self, input_embdings: torch.Tensor, infer_state, layer_weight):
        if infer_state.is_prefill:
            # Commit prefill only writes draft KV; BaseModel still requires a tensor output.
            return input_embdings.new_empty((0,))
        return super().token_forward(
            input_embdings=input_embdings,
            infer_state=infer_state,
            layer_weight=layer_weight,
        )
