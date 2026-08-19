import torch

from lightllm.models.neo_chat_moe.infer_struct import NeoChatInferStateInfo
from lightllm.models.qwen3.layer_infer.transformer_layer_infer import Qwen3TransformerLayerInfer
from lightllm.models.neo_chat.layer_weights.transformer_layer_weight import NeoChatTransformerLayerWeight
from lightllm.common.basemodel.attention.base_att import AttControl


class NeoChatTransformerLayerInfer(Qwen3TransformerLayerInfer):
    def __init__(self, layer_num, network_config):
        super().__init__(layer_num, network_config)

    def _get_qkv(self, input, infer_state: NeoChatInferStateInfo, layer_weight: NeoChatTransformerLayerWeight):
        input = input.view(-1, self.embed_dim_)
        input = self._tpsp_allgather(input=input, infer_state=infer_state)

        qkv = layer_weight.qkv_proj.mm(input)
        q, cache_kv = qkv.split(
            [self.tp_q_head_num_ * self.head_dim_, (self.tp_k_head_num_ + self.tp_v_head_num_) * self.head_dim_], dim=-1
        )
        q = q.view(q.shape[0], self.tp_q_head_num_, self.head_dim_)
        q_t, q_hw = q.chunk(2, dim=-1)

        cache_kv = cache_kv.view(-1, (self.tp_k_head_num_ + self.tp_v_head_num_), self.head_dim_)
        k = cache_kv[:, : self.tp_k_head_num_, :]
        v = cache_kv[:, self.tp_k_head_num_ :, :]
        k_t, k_hw = k.chunk(2, dim=-1)

        q_t_2d = q_t.reshape(q.shape[0], -1)
        q_hw_2d = q_hw.reshape(q.shape[0], -1)
        k_t_2d = k_t.reshape(k.shape[0], -1)
        k_hw_2d = k_hw.reshape(k.shape[0], -1)

        layer_weight.qk_norm_weight_(q_t_2d, k_t_2d, eps=self.eps_)
        layer_weight.qk_hw_norm_weight_(q_hw_2d, k_hw_2d, eps=self.eps_)

        q_t = q_t_2d.view(q.shape[0], self.tp_q_head_num_, self.head_dim_ // 2)
        q_hw = q_hw_2d.view(q.shape[0], self.tp_q_head_num_, self.head_dim_ // 2)
        q_h, q_w = (part.contiguous() for part in q_hw.chunk(2, dim=-1))

        k_t = k_t_2d.view(k.shape[0], self.tp_k_head_num_, self.head_dim_ // 2)
        k_hw = k_hw_2d.view(k.shape[0], self.tp_k_head_num_, self.head_dim_ // 2)
        k_h, k_w = (part.contiguous() for part in k_hw.chunk(2, dim=-1))

        self.platform_backend.ops.rotary_emb(
            is_prefill=infer_state.is_prefill,
            batch_size=infer_state.batch_size,
            q=q_t,
            k=k_t,
            cos=infer_state.position_cos,
            sin=infer_state.position_sin,
        )
        self.platform_backend.ops.rotary_emb(
            is_prefill=infer_state.is_prefill,
            batch_size=infer_state.batch_size,
            q=q_h,
            k=k_h,
            cos=infer_state.position_cos_h,
            sin=infer_state.position_sin_h,
        )
        self.platform_backend.ops.rotary_emb(
            is_prefill=infer_state.is_prefill,
            batch_size=infer_state.batch_size,
            q=q_w,
            k=k_w,
            cos=infer_state.position_cos_w,
            sin=infer_state.position_sin_w,
        )

        q = torch.cat([q_t, q_h, q_w], dim=-1)
        q = q.reshape(q.shape[0], -1)

        k = torch.cat([k_t, k_h, k_w], dim=-1)
        cache_kv = torch.cat([k, v], dim=1)

        if infer_state.need_dp_prefill_balance:
            q = infer_state._all_to_all_unbalance_get(data=q)
            cache_kv = infer_state._all_to_all_unbalance_get(data=cache_kv)
        return q, cache_kv

    def _context_attention_kernel(
        self, q, kv, infer_state: NeoChatInferStateInfo, layer_weight, out=None
    ) -> torch.Tensor:

        _q = q.view(-1, self.tp_q_head_num_, self.head_dim_)
        _k, _v = infer_state.mem_manager.get_att_input_params(layer_index=self.layer_num_)

        att_control = AttControl()
        att_control.image_token_tag = getattr(infer_state, "b_image_token_tag", None)

        o_tensor = infer_state.prefill_att_state.prefill_att(
            q=_q,
            k=_k,
            v=_v,
            att_control=att_control,
            alloc_func=self.alloc_tensor,
        )
        return o_tensor.view(q.shape)

    def _token_attention_kernel(
        self,
        q: torch.Tensor,
        infer_state: NeoChatInferStateInfo,
        layer_weight: NeoChatTransformerLayerWeight,
    ) -> torch.Tensor:
        _k, _v = infer_state.mem_manager.get_att_input_params(layer_index=self.layer_num_)
        _q = q.view(-1, self.tp_q_head_num_, self.head_dim_)
        att_control = AttControl()
        o_tensor = infer_state.decode_att_state.decode_att(
            q=_q, k=_k, v=_v, att_control=att_control, alloc_func=self.alloc_tensor
        )
        return o_tensor.view(q.shape)
