from lightllm.models.glm5_next.layer_infer.transformer_layer_infer import Glm5NextTransformerLayerInfer


class Glm5NextMTPTransformerLayerInfer(Glm5NextTransformerLayerInfer):
    def context_forward(self, input_embeddings, infer_state, layer_weight):
        if not getattr(infer_state, "prefill_kv_only", False):
            return super().context_forward(input_embeddings, infer_state, layer_weight)

        normed = self._att_norm(input_embeddings, infer_state, layer_weight)
        hidden, _, kv = self._project_sparse_qkv_a(normed, infer_state, layer_weight)
        infer_state.mtp_draft_input_hiddens = None
        self._post_cache_kv(self._normalize_sparse_kv(kv, layer_weight), infer_state, layer_weight)
        self.indexer.write_k_pools(
            hidden, infer_state, self.prefill_attention.context(infer_state.attention), layer_weight
        )
        # BaseModel pads empty ranks to at least one row. An empty result marks
        # the discarded draft output for our post layer without retaining hiddens.
        return input_embeddings.new_empty((0, self.embed_dim_))

    def overlap_tpsp_context_forward(
        self, input_embeddings, input_embeddings1, infer_state, infer_state1, layer_weight
    ):
        return (
            self.context_forward(input_embeddings, infer_state, layer_weight),
            self.context_forward(input_embeddings1, infer_state1, layer_weight),
        )
