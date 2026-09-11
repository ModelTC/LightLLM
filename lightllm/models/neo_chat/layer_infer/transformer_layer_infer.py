import torch

from lightllm.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.models.neo_chat.triton_kernel.context_attention_fwd_neo import (
    context_attention_fwd_neo,
)
from lightllm.models.qwen3.layer_infer.transformer_layer_infer import (
    Qwen3TransformerLayerInfer,
)


class NeoChatTransformerLayerInfer(Qwen3TransformerLayerInfer):
    """Qwen3 transformer layer with Neo's image-aware attention."""

    def _get_qkv(self, input, infer_state, layer_weight):
        input = input.view(-1, self.embed_dim_)
        input = self._tpsp_allgather(input=input, infer_state=infer_state)

        qkv = layer_weight.qkv_proj.mm(input)
        q_width = self.tp_q_head_num_ * self.head_dim_
        kv_width = (self.tp_k_head_num_ + self.tp_v_head_num_) * self.head_dim_
        q, cache_kv = qkv.split([q_width, kv_width], dim=-1)

        q = q.view(-1, self.tp_q_head_num_, self.head_dim_)
        cache_kv = cache_kv.view(
            -1,
            self.tp_k_head_num_ + self.tp_v_head_num_,
            self.head_dim_,
        )
        k = cache_kv[:, : self.tp_k_head_num_]
        v = cache_kv[:, self.tp_k_head_num_ :]

        # Neo applies independent Q/K normalization and RoPE to temporal,
        # height, and width dimensions.
        q_t, q_hw = q.chunk(2, dim=-1)
        k_t, k_hw = k.chunk(2, dim=-1)
        q_t = q_t.reshape(q.shape[0], -1)
        q_hw = q_hw.reshape(q.shape[0], -1)
        k_t = k_t.reshape(k.shape[0], -1)
        k_hw = k_hw.reshape(k.shape[0], -1)
        layer_weight.qk_norm_weight_(q_t, k_t, eps=self.eps_)
        layer_weight.qk_hw_norm_weight_(q_hw, k_hw, eps=self.eps_)

        q_t = q_t.view(q.shape[0], self.tp_q_head_num_, self.head_dim_ // 2)
        q_h, q_w = q_hw.view(
            q.shape[0], self.tp_q_head_num_, self.head_dim_ // 2
        ).chunk(2, dim=-1)
        k_t = k_t.view(k.shape[0], self.tp_k_head_num_, self.head_dim_ // 2)
        k_h, k_w = k_hw.view(
            k.shape[0], self.tp_k_head_num_, self.head_dim_ // 2
        ).chunk(2, dim=-1)

        rotary_emb_fwd(q_t, k_t, infer_state.position_cos, infer_state.position_sin)
        rotary_emb_fwd(q_h, k_h, infer_state.position_cos_h, infer_state.position_sin_h)
        rotary_emb_fwd(q_w, k_w, infer_state.position_cos_w, infer_state.position_sin_w)

        q = torch.cat((q_t, q_h, q_w), dim=-1).reshape(q.shape[0], -1)
        k = torch.cat((k_t, k_h, k_w), dim=-1)
        cache_kv = torch.cat((k, v), dim=1)

        if infer_state.need_dp_prefill_balance:
            q = infer_state._all_to_all_unbalance_get(data=q)
            cache_kv = infer_state._all_to_all_unbalance_get(data=cache_kv)
        return q, cache_kv

    def _context_attention_kernel(self, q, kv, infer_state, layer_weight, out=None):
        output = self.alloc_tensor(q.shape, q.dtype) if out is None else out
        k_pool, v_pool = infer_state.mem_manager.get_att_input_params(
            layer_index=self.layer_num_
        )
        context_attention_fwd_neo(
            q.view(-1, self.tp_q_head_num_, self.head_dim_),
            k_pool,
            v_pool,
            output.view(-1, self.tp_q_head_num_, self.head_dim_),
            infer_state.b_req_idx,
            infer_state.b_q_start_loc,
            infer_state.b_seq_len,
            infer_state.b_ready_cache_len,
            infer_state.max_q_seq_len,
            infer_state.req_manager.req_to_token_indexs,
            infer_state.b_image_token_end,
        )
        return output
