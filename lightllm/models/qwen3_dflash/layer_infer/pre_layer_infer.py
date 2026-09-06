from lightllm.models.llama.layer_infer.pre_layer_infer import LlamaPreLayerInfer
from lightllm.models.qwen3_dflash.layer_weights.pre_and_post_layer_weight import Qwen3DFlashPreAndPostLayerWeight


class Qwen3DFlashPreLayerInfer(LlamaPreLayerInfer):
    """Project target hiddens for DFlash commit prefill."""

    def __init__(self, network_config):
        super().__init__(network_config)
        self.eps_ = network_config["rms_norm_eps"]

    def context_forward(
        self,
        input_ids,
        infer_state,
        layer_weight: Qwen3DFlashPreAndPostLayerWeight,
    ):
        target_hidden_states = layer_weight.fc_weight_.mm(
            infer_state.mtp_draft_input_hiddens,
            use_custom_tensor_mananger=False,
        )
        return layer_weight.hidden_norm_weight_(
            input=target_hidden_states,
            eps=self.eps_,
            alloc_func=self.alloc_tensor,
        )
