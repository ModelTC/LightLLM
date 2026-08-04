import torch

from lightllm.models.llama.layer_infer.post_layer_infer import LlamaPostLayerInfer


class Qwen3DFlashPostLayerInfer(LlamaPostLayerInfer):
    def _is_commit_prefill(self, infer_state):
        return infer_state.is_prefill and infer_state.mtp_draft_input_hiddens is not None

    def _tpsp_allgather(self, input: torch.Tensor, infer_state):
        if self._is_commit_prefill(infer_state):
            return input
        return super()._tpsp_allgather(input=input, infer_state=infer_state)

    def token_forward(self, input_embdings: torch.Tensor, infer_state, layer_weight):
        if self._is_commit_prefill(infer_state):
            # Commit prefill only materializes draft KV.  There is no LM head
            # work to do, but BaseModel still expects a logits-shaped tensor.
            return torch.empty(
                (infer_state.input_ids.shape[0], 0),
                dtype=input_embdings.dtype,
                device=input_embdings.device,
            )
        return super().token_forward(
            input_embdings=input_embdings,
            infer_state=infer_state,
            layer_weight=layer_weight,
        )
