import torch

from lightllm.models.llama.layer_infer.pre_layer_infer import LlamaPreLayerInfer
from lightllm.models.qwen3_dflash.layer_weights.pre_and_post_layer_weight import Qwen3DFlashPreAndPostLayerWeight


class Qwen3DFlashPreLayerInfer(LlamaPreLayerInfer):
    """DFlash target-hidden projection plus normal token embedding."""

    def __init__(self, network_config):
        super().__init__(network_config)
        self.eps_ = network_config["rms_norm_eps"]
        self.hidden_size_ = network_config["hidden_size"]
        return

    def project_target_hidden(
        self,
        *,
        target_hidden_states: torch.Tensor,
        layer_weight: Qwen3DFlashPreAndPostLayerWeight,
    ) -> torch.Tensor:
        if target_hidden_states.dim() == 2:
            batch_size = target_hidden_states.shape[0]
            context_len = 1
            flat_hidden = target_hidden_states
        else:
            assert target_hidden_states.dim() == 3
            batch_size, context_len, _ = target_hidden_states.shape
            flat_hidden = target_hidden_states.reshape(batch_size * context_len, -1)

        projected = layer_weight.fc_weight_.mm(flat_hidden, use_custom_tensor_mananger=False)
        projected = layer_weight.hidden_norm_weight_(
            input=projected,
            eps=self.eps_,
            alloc_func=self.alloc_tensor,
        )
        return projected.view(batch_size, context_len, self.hidden_size_)

    def context_forward(
        self,
        input_ids,
        infer_state,
        layer_weight: Qwen3DFlashPreAndPostLayerWeight,
    ):
        if infer_state.mtp_draft_input_hiddens is None:
            return super().context_forward(input_ids, infer_state, layer_weight)

        return self.project_target_hidden(
            target_hidden_states=infer_state.mtp_draft_input_hiddens,
            layer_weight=layer_weight,
        ).reshape(-1, self.hidden_size_)

    def token_forward(
        self,
        input_ids,
        infer_state,
        layer_weight: Qwen3DFlashPreAndPostLayerWeight,
    ):
        return super().token_forward(input_ids, infer_state, layer_weight)

    def decode_forward(
        self,
        input_ids,
        infer_state,
        layer_weight: Qwen3DFlashPreAndPostLayerWeight,
    ):
        return self.token_forward(input_ids, infer_state, layer_weight)
