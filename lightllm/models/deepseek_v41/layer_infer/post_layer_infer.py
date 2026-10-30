from lightllm.models.llama.layer_infer.post_layer_infer import LlamaPostLayerInfer


class DeepseekV41PostLayerInfer(LlamaPostLayerInfer):
    def token_forward(self, input_embdings, infer_state, layer_weight):
        if infer_state.is_prefill and infer_state.decoder_state is not None:
            infer_state = infer_state.decoder_state
        streams, pre_mix = input_embdings
        streams = streams.view(streams.shape[0], pre_mix.shape[1], -1)
        collapsed = (streams.float() * pre_mix[..., None]).sum(1).to(streams.dtype)
        return super().token_forward(collapsed, infer_state, layer_weight)
