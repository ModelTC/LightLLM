from lightllm.common.basemodel.triton_kernel.mhc import hc_combine
from lightllm.models.llama.layer_infer.post_layer_infer import LlamaPostLayerInfer


class DeepseekV41PostLayerInfer(LlamaPostLayerInfer):
    def _slice_get_last_input(self, input_embdings, infer_state):
        streams, pre_mix = input_embdings
        if infer_state.is_prefill and not infer_state.return_all_prompt_logics:
            last_input = hc_combine(streams, pre_mix, cu_q_lens=infer_state.b1_cu_q_seq_len)
            return last_input, infer_state.batch_size
        return super()._slice_get_last_input(hc_combine(streams, pre_mix), infer_state)
