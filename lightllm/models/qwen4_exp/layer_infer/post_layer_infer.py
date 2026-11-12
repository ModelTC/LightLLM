from lightllm.models.llama.layer_infer.post_layer_infer import LlamaPostLayerInfer


class Qwen4ExpPostLayerInfer(LlamaPostLayerInfer):
    def _norm(self, input, infer_state, layer_weight):
        return layer_weight.hyper_connection_mixer.component.read(input)
