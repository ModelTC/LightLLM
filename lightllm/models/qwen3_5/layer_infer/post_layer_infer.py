from lightllm.models.llama.layer_infer.post_layer_infer import LlamaPostLayerInfer


class Qwen35PostLayerInfer(LlamaPostLayerInfer):
    def token_forward(self, input_embdings, infer_state, layer_weight):
        normed = self._norm(input_embdings, infer_state, layer_weight)
        output = super().token_forward(normed, infer_state, layer_weight)
        output.final_hidden = normed
        return output

    def _project_local_logits(self, hidden, token_num, layer_weight, infer_state):
        hidden = hidden.permute(1, 0).view(-1, token_num)
        return layer_weight.lm_head_weight_(input=hidden, alloc_func=self.alloc_tensor)
