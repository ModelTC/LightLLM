from lightllm.models.qwen_vl.layer_infer.pre_layer_infer import LlamaMultimodalPreLayerInfer


class Glm5NextMTPPreLayerInfer(LlamaMultimodalPreLayerInfer):
    """Resolve shifted image tokens before the standard NextN embedding/hidden fusion."""

    def __init__(self, network_config):
        super().__init__(network_config)
        self.eps_ = network_config["rms_norm_eps"]

    def _fuse_hidden(self, input_embeddings, infer_state, layer_weight):
        previous_hidden = infer_state.mtp_draft_input_hiddens
        from lightllm.models.glm5_next_mtp.triton_kernel.fused_eh_norm import fused_mtp_norm

        normalized_embeddings = fused_mtp_norm(
            embeddings=input_embeddings,
            previous_hidden=previous_hidden,
            main_weight=layer_weight.main_norm_weight_.weight,
            embedding_weight=layer_weight.enorm_weight_.weight,
            hidden_weight=layer_weight.hnorm_weight_.weight,
            eps=self.eps_,
        )
        return layer_weight.eh_proj_weight_.mm(input_tensor=normalized_embeddings)

    def context_forward(self, input_ids, infer_state, layer_weight):
        input_embeddings = super().context_forward(input_ids, infer_state, layer_weight)
        return self._fuse_hidden(input_embeddings, infer_state, layer_weight)

    def token_forward(self, input_ids, infer_state, layer_weight):
        input_embeddings = super().token_forward(input_ids, infer_state, layer_weight)
        return self._fuse_hidden(input_embeddings, infer_state, layer_weight)
