from lightllm.common.basemodel.triton_kernel.mhc import hc_expand
from lightllm.models.llama.layer_infer.pre_layer_infer import LlamaPreLayerInfer
from lightllm.models.qwen_vl.layer_infer.pre_layer_infer import LlamaMultimodalPreLayerInfer
from ..infer_struct import DeepseekV4InferStateInfo


class DeepseekV4PreLayerInfer(LlamaMultimodalPreLayerInfer):
    """Token embedding, then expand to the hc_mult parallel residual streams [T, hc_mult*hidden]."""

    def __init__(self, network_config):
        super().__init__(network_config)
        self.hc_mult = network_config["hc_mult"]
        self.has_vision = network_config.get("vision_n_layers", 0) > 0
        return

    def context_forward(self, input_ids, infer_state: DeepseekV4InferStateInfo, layer_weight):
        if self.has_vision:
            input_embdings = super().context_forward(input_ids, infer_state, layer_weight)
        else:
            input_embdings = LlamaPreLayerInfer.context_forward(self, input_ids, infer_state, layer_weight)
        infer_state.mtp_draft_input_hiddens = None
        return hc_expand(input_embdings, self.hc_mult)

    def token_forward(self, input_ids, infer_state: DeepseekV4InferStateInfo, layer_weight):
        input_embdings = super().token_forward(input_ids, infer_state, layer_weight)
        if not infer_state.is_cuda_graph:
            infer_state.mtp_draft_input_hiddens = None
        return hc_expand(input_embdings, self.hc_mult)
