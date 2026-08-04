from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.models.llama.layer_infer.pre_layer_infer import LlamaPreLayerInfer
from lightllm.models.qwen3_eagle.layer_weights.pre_and_post_layer_weight import Qwen3EaglePreAndPostLayerWeight


class Qwen3EaglePreLayerInfer(LlamaPreLayerInfer):
    """Eagle3 draft-token embedding plus target-hidden projection."""

    def __init__(self, network_config):
        super().__init__(network_config)
        self.hidden_size_ = network_config["hidden_size"]
        return

    def prepare_mtp_draft_hiddens(
        self,
        infer_state: InferStateInfo,
        layer_weight: Qwen3EaglePreAndPostLayerWeight,
    ) -> None:
        # Keep the ModelInput hidden raw for CUDA graph replay; Eagle layers consume this working buffer.
        infer_state.eagle_draft_hidden_states = self.project_mtp_draft_hiddens(
            infer_state.mtp_draft_input_hiddens,
            layer_weight,
        )
        return

    def project_mtp_draft_hiddens(
        self,
        target_hiddens,
        layer_weight: Qwen3EaglePreAndPostLayerWeight,
        use_custom_tensor_mananger: bool = True,
    ):
        if target_hiddens is None or target_hiddens.shape[-1] == self.hidden_size_:
            return target_hiddens
        return layer_weight.fc_weight_.mm(
            target_hiddens,
            use_custom_tensor_mananger=use_custom_tensor_mananger,
        )

    def context_forward(
        self,
        input_ids,
        infer_state: InferStateInfo,
        layer_weight: Qwen3EaglePreAndPostLayerWeight,
    ):
        self.prepare_mtp_draft_hiddens(infer_state, layer_weight)
        return super().context_forward(input_ids, infer_state, layer_weight)

    def token_forward(
        self,
        input_ids,
        infer_state: InferStateInfo,
        layer_weight: Qwen3EaglePreAndPostLayerWeight,
    ):
        self.prepare_mtp_draft_hiddens(infer_state, layer_weight)
        return super().token_forward(input_ids, infer_state, layer_weight)
