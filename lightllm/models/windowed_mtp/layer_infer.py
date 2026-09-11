from lightllm.utils.sgl_utils import flash_attn_with_kvcache


class WindowedAttentionMixin:
    def token_attention_forward(self, input_embdings, infer_state, layer_weight):
        q, noise_kv = self._get_qkv(input_embdings, infer_state, layer_weight)
        store, reqs, counts, block = infer_state.windowed_context
        scratch = store.pack(self.windowed_layer_index, reqs, noise_kv, block)
        output = flash_attn_with_kvcache(
            q.view(-1, block, self.tp_q_head_num_, self.head_dim_),
            scratch[:, :, : self.tp_k_head_num_],
            scratch[:, :, self.tp_k_head_num_ :],
            cache_seqlens=counts + block,
            causal=False,
            softmax_scale=self.head_dim_ ** -0.5,
        )
        return self._get_o(output.reshape(-1, self.tp_q_head_num_ * self.head_dim_), infer_state, layer_weight)
