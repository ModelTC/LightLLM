from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.models.llama.layer_infer.pre_layer_infer import LlamaPreLayerInfer
from lightllm.models.qwen3_eagle.layer_weights.pre_and_post_layer_weight import Qwen3EaglePreAndPostLayerWeight


class Qwen3EaglePreLayerInfer(LlamaPreLayerInfer):
    """EAGLE3 draft-token embedding and fixed-width draft hidden preparation."""

    def __init__(self, network_config):
        super().__init__(network_config)
        self.hidden_size_ = network_config["hidden_size"]

    def prepare_spec_draft_hiddens(
        self,
        infer_state: InferStateInfo,
    ) -> None:
        draft_hiddens = infer_state.mtp_draft_input_hiddens
        assert draft_hiddens is not None
        assert draft_hiddens.shape[-1] == self.hidden_size_
        infer_state.eagle_draft_hidden_states = draft_hiddens

    def context_forward(
        self,
        input_ids,
        infer_state: InferStateInfo,
        layer_weight: Qwen3EaglePreAndPostLayerWeight,
    ):
        self.prepare_spec_draft_hiddens(infer_state)
        return super().context_forward(input_ids, infer_state, layer_weight)

    def token_forward(
        self,
        input_ids,
        infer_state: InferStateInfo,
        layer_weight: Qwen3EaglePreAndPostLayerWeight,
    ):
        self.prepare_spec_draft_hiddens(infer_state)
        return super().token_forward(input_ids, infer_state, layer_weight)
