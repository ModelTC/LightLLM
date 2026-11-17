from lightllm.models.qwen3_vl.layer_infer.pre_layer_infer import Qwen3VLMultimodalPreLayerInfer


class SenseNovaLoopPreLayerInfer(Qwen3VLMultimodalPreLayerInfer):
    inline_vision = None

    def context_forward(self, input_ids, infer_state, layer_weight):
        if self.inline_vision is not None:
            self.inline_vision.ensure_embeddings(infer_state.multimodal_params)
        return super().context_forward(input_ids, infer_state, layer_weight)
