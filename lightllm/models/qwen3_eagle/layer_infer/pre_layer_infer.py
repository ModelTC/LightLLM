from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.models.llama.layer_infer.pre_layer_infer import LlamaPreLayerInfer
from lightllm.models.qwen3_eagle.layer_weights.pre_and_post_layer_weight import Qwen3EaglePreAndPostLayerWeight


class Qwen3EaglePreLayerInfer(LlamaPreLayerInfer):
    """Eagle3 draft-token embedding plus target-hidden projection."""

    def __init__(self, network_config):
        super().__init__(network_config)
        self.hidden_size_ = network_config["hidden_size"]

    def prepare_spec_draft_hiddens(
        self,
        infer_state: InferStateInfo,
        layer_weight: Qwen3EaglePreAndPostLayerWeight,
    ) -> None:
        target_hiddens = infer_state.mtp_draft_input_hiddens
        # Target verification provides concatenated auxiliary-layer hiddens (N * H).
        # Autoregressive draft steps feed the previous draft output, which is already H.
        if target_hiddens.shape[-1] != self.hidden_size_:
            target_hiddens = layer_weight.fc_weight_.mm(target_hiddens)
        infer_state.eagle_draft_hidden_states = target_hiddens

    def context_forward(
        self,
        input_ids,
        infer_state: InferStateInfo,
        layer_weight: Qwen3EaglePreAndPostLayerWeight,
    ):
        self.prepare_spec_draft_hiddens(infer_state, layer_weight)
        return super().context_forward(input_ids, infer_state, layer_weight)

    def token_forward(
        self,
        input_ids,
        infer_state: InferStateInfo,
        layer_weight: Qwen3EaglePreAndPostLayerWeight,
    ):
        self.prepare_spec_draft_hiddens(infer_state, layer_weight)
        return super().token_forward(input_ids, infer_state, layer_weight)
