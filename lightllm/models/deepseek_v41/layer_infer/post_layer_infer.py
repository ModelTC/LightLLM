from lightllm.common.basemodel.triton_kernel.mhc import hc_combine
from lightllm.models.llama.layer_infer.post_layer_infer import LlamaPostLayerInfer


class DeepseekV41PostLayerInfer(LlamaPostLayerInfer):
    def token_forward(self, input_embdings, infer_state, layer_weight):
        if infer_state.is_prefill and infer_state.decoder_state is not None:
            infer_state = infer_state.decoder_state
        streams, pre_mix = input_embdings
        collapsed = hc_combine(streams, pre_mix)
        return super().token_forward(collapsed, infer_state, layer_weight)
