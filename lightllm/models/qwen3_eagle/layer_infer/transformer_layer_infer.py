import torch

from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.models.qwen3_eagle.layer_weights.transformer_layer_weight import Qwen3EagleTransformerLayerWeight


class Qwen3EagleTransformerLayerInfer(LlamaTransformerLayerInfer):
    def __init__(self, layer_num, network_config):
        super().__init__(layer_num, network_config)
        self.head_dim_ = network_config["head_dim"]
        return

    def _get_qkv(self, input, infer_state: InferStateInfo, layer_weight: Qwen3EagleTransformerLayerWeight):
        input_part = self._att_norm(input, infer_state, layer_weight)
        target_part = layer_weight.hidden_norm_weight_._native_forward(
            input=infer_state.eagle_draft_hidden_states,
            eps=self.eps_,
            alloc_func=self.alloc_tensor,
        )
        input = torch.cat([input_part, target_part], dim=-1)
        input = input.view(-1, self.embed_dim_ * 2)
        input = self._tpsp_allgather(input, infer_state)
        q = layer_weight.q_proj.mm(input)
        cache_kv = layer_weight.kv_proj.mm(input)
        layer_weight.qk_norm_weight_(
            q,
            cache_kv[:, : self.tp_k_head_num_ * self.head_dim_],
            eps=self.eps_,
        )
        cache_kv = cache_kv.view(-1, (self.tp_k_head_num_ + self.tp_v_head_num_), self.head_dim_)
        rotary_emb_fwd(
            q.view(-1, self.tp_q_head_num_, self.head_dim_),
            cache_kv[:, : self.tp_k_head_num_, :],
            infer_state.position_cos,
            infer_state.position_sin,
        )

        if infer_state.need_dp_prefill_balance:
            q = infer_state._all_to_all_unbalance_get(data=q)
            cache_kv = infer_state._all_to_all_unbalance_get(data=cache_kv)

        return q, cache_kv

    def context_forward(self, input_embdings, infer_state: InferStateInfo, layer_weight):
        o = self.context_attention_forward(input_embdings, infer_state, layer_weight)

        hidden_states = infer_state.eagle_draft_hidden_states + o.view(-1, self.embed_dim_)
        ffn_input = self._ffn_norm(hidden_states, infer_state, layer_weight)
        ffn_out = self._ffn(ffn_input, infer_state, layer_weight)

        hidden_states = hidden_states + ffn_out.view(-1, self.embed_dim_)
        infer_state.eagle_draft_hidden_states = hidden_states
        return hidden_states

    def token_forward(self, input_embdings, infer_state: InferStateInfo, layer_weight):
        o = self.token_attention_forward(input_embdings, infer_state, layer_weight)

        hidden_states = infer_state.eagle_draft_hidden_states + o.view(-1, self.embed_dim_)
        ffn_input = self._ffn_norm(hidden_states, infer_state, layer_weight)
        ffn_out = self._ffn(ffn_input, infer_state, layer_weight)

        hidden_states = hidden_states + ffn_out.view(-1, self.embed_dim_)
        infer_state.eagle_draft_hidden_states = hidden_states
        return hidden_states
