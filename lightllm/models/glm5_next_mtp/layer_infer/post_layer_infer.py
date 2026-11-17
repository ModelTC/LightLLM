from lightllm.common.basemodel.batch_objs import PostLayerOutput
from lightllm.common.basemodel.hidden_collector import NoopHiddenCollector
from lightllm.models.llama.layer_infer.post_layer_infer import LlamaPostLayerInfer


class Glm5NextMTPPostLayerInfer(LlamaPostLayerInfer):
    def token_forward(self, input_embeddings, infer_state, layer_weight):
        if infer_state.is_prefill and input_embeddings.shape[0] == 0:
            infer_state.hidden_collector = NoopHiddenCollector()
            return PostLayerOutput(logits=input_embeddings.new_empty((0,)))
        return super().token_forward(input_embeddings, infer_state, layer_weight)
