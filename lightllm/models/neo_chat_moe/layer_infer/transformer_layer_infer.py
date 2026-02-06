import torch
from functools import partial
from typing import Tuple
from lightllm.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.models.neo_chat_moe.infer_struct import NeoChatInferStateInfo
from lightllm.models.neo_chat_moe.triton_kernel.context_attention_fwd_neo import context_attention_fwd_neo
from lightllm.models.llama.triton_kernel.token_attention_nopad_att1 import token_att_fwd
from lightllm.models.qwen3_moe.layer_infer.transformer_layer_infer import Qwen3MOETransformerLayerInfer
from lightllm.models.neo_chat_moe.layer_weights.transformer_layer_weight import NeoChatMOETransformerLayerWeight
from lightllm.distributed import all_reduce
import torch.distributed as dist
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.common.basemodel.attention.base_att import AttControl


class NeoChatMOETransformerLayerInfer(Qwen3MOETransformerLayerInfer):
    def __init__(self, data_type, network_config):
        self._is_merge_kv = network_config.get("merge_kv", True)
        super().__init__(data_type, network_config)
        return

    def _bind_attention(self):
        self._context_attention_kernel = self._context_attention_kernel
        self._token_attention_kernel = self._token_decode_attention_normal
        self._copy_kv_to_mem_cache = self._copy_kv_to_mem_cache_normal
        return

    def _get_qkv(self, input, infer_state: NeoChatInferStateInfo, layer_weight: NeoChatMOETransformerLayerWeight):
        if self._is_merge_kv:
            return self._get_qkv_mergekv(input, infer_state, layer_weight)
        else:
            return self._get_qkv_not_mergekv(input, infer_state, layer_weight)

    def _get_qkv_not_mergekv(
        self, input, infer_state: NeoChatInferStateInfo, layer_weight: NeoChatMOETransformerLayerWeight
    ):
        input = input.view(-1, self.embed_dim_)
        q = layer_weight.q_proj.mm(input)  # [T, Hq*D]

        q_hw = layer_weight.q_hw_proj.mm(input)
        q_hw = q_hw.view(-1, self.tp_q_head_num_, self.head_dim_)
        q_h, q_w = q_hw.chunk(2, dim=-1)

        k_hw = layer_weight.k_hw_proj.mm(input)
        k_hw = k_hw.view(-1, self.tp_k_head_num_, self.head_dim_)
        k_h, k_w = k_hw.chunk(2, dim=-1)

        cache_kv = layer_weight.kv_proj.mm(input)  # [T, (Hk+Hv)*D]

        layer_weight.q_norm_weight_(q, eps=self.eps_)

        q_h_2d = q_h.reshape(q.shape[0], -1)
        q_w_2d = q_w.reshape(q.shape[0], -1)
        layer_weight.q_norm_h_weight_(q_h_2d, eps=self.eps_)
        layer_weight.q_norm_w_weight_(q_w_2d, eps=self.eps_)
        q_h = q_h_2d.view(q.shape[0], self.tp_q_head_num_, self.head_dim_ // 2)
        q_w = q_w_2d.view(q.shape[0], self.tp_q_head_num_, self.head_dim_ // 2)

        layer_weight.k_norm_weight_(
            cache_kv[:, : self.tp_k_head_num_ * self.head_dim_],
            eps=self.eps_,
        )

        k_h_2d = k_h.reshape(q.shape[0], -1)  # [T, Hk*(D/2)]
        k_w_2d = k_w.reshape(q.shape[0], -1)
        layer_weight.k_norm_h_weight_(k_h_2d, eps=self.eps_)
        layer_weight.k_norm_w_weight_(k_w_2d, eps=self.eps_)
        k_h = k_h_2d.view(q.shape[0], self.tp_k_head_num_, self.head_dim_ // 2)
        k_w = k_w_2d.view(q.shape[0], self.tp_k_head_num_, self.head_dim_ // 2)

        cache_kv = cache_kv.view(-1, (self.tp_k_head_num_ + self.tp_v_head_num_), self.head_dim_)

        rotary_emb_fwd(
            q.view(-1, self.tp_q_head_num_, self.head_dim_),
            cache_kv[:, : self.tp_k_head_num_, :],
            infer_state.position_cos,
            infer_state.position_sin,
        )
        rotary_emb_fwd(
            q_h,
            k_h,
            infer_state.position_cos_h,
            infer_state.position_sin_h,
        )
        rotary_emb_fwd(
            q_w,
            k_w,
            infer_state.position_cos_w,
            infer_state.position_sin_w,
        )

        q3 = q.view(-1, self.tp_q_head_num_, self.head_dim_)
        q3 = torch.cat([q3, q_h, q_w], dim=-1)
        q = q3.reshape(q3.shape[0], -1)

        k = cache_kv[:, : self.tp_k_head_num_, :]
        k = torch.cat([k, k_h, k_w], dim=-1)

        v = cache_kv[:, self.tp_k_head_num_ : self.tp_k_head_num_ + self.tp_v_head_num_, :]
        v_pad = torch.zeros((v.shape[0], v.shape[1], self.head_dim_), device=v.device, dtype=v.dtype)
        v = torch.cat([v, v_pad], dim=-1)

        cache_kv = torch.cat([k, v], dim=1)
        return q, cache_kv

    def _get_qkv_mergekv(
        self, input, infer_state: NeoChatInferStateInfo, layer_weight: NeoChatMOETransformerLayerWeight
    ):
        input = input.view(-1, self.embed_dim_)

        qkv = layer_weight.qkv_proj.mm(input)
        q, cache_kv = qkv.split(
            [self.tp_q_head_num_ * self.head_dim_, (self.tp_k_head_num_ + self.tp_v_head_num_) * self.head_dim_], dim=-1
        )
        q_hw = layer_weight.q_hw_proj.mm(input)
        k_hw = layer_weight.k_hw_proj.mm(input)

        layer_weight.q_norm_weight_(q, eps=self.eps_)
        layer_weight.q_norm_hw_weight_(q_hw, eps=self.eps_)
        layer_weight.k_norm_hw_weight_(k_hw, eps=self.eps_)

        q_hw = q_hw.view(q.shape[0], self.tp_q_head_num_, self.head_dim_)
        q_h, q_w = q_hw.chunk(2, dim=-1)

        layer_weight.k_norm_weight_(
            cache_kv[:, : self.tp_k_head_num_ * self.head_dim_],
            eps=self.eps_,
        )

        k_hw = k_hw.view(q.shape[0], self.tp_k_head_num_, self.head_dim_)
        k_h, k_w = k_hw.chunk(2, dim=-1)

        cache_kv = cache_kv.view(-1, (self.tp_k_head_num_ + self.tp_v_head_num_), self.head_dim_)

        rotary_emb_fwd(
            q.view(-1, self.tp_q_head_num_, self.head_dim_),
            cache_kv[:, : self.tp_k_head_num_, :],
            infer_state.position_cos,
            infer_state.position_sin,
        )
        rotary_emb_fwd(
            q_h,
            k_h,
            infer_state.position_cos_h,
            infer_state.position_sin_h,
        )
        rotary_emb_fwd(
            q_w,
            k_w,
            infer_state.position_cos_w,
            infer_state.position_sin_w,
        )

        q3 = q.view(-1, self.tp_q_head_num_, self.head_dim_)
        q3 = torch.cat([q3, q_h, q_w], dim=-1)
        q = q3.reshape(q3.shape[0], -1)

        k = cache_kv[:, : self.tp_k_head_num_, :]
        k = torch.cat([k, k_h, k_w], dim=-1)

        v = cache_kv[:, self.tp_k_head_num_ : self.tp_k_head_num_ + self.tp_v_head_num_, :]
        v_pad = torch.zeros((v.shape[0], v.shape[1], self.head_dim_), device=v.device, dtype=v.dtype)
        v = torch.cat([v, v_pad], dim=-1)

        cache_kv = torch.cat([k, v], dim=1)
        return q, cache_kv

    def _context_attention_kernel(
        self, q, kv, infer_state: NeoChatInferStateInfo, layer_weight, out=None
    ) -> torch.Tensor:
        o_tensor = self.alloc_tensor(q.shape, q.dtype) if out is None else out
        kv = infer_state.mem_manager.kv_buffer[self.layer_num_]
        context_attention_fwd_neo(
            q.view(-1, self.tp_q_head_num_, self.head_dim_ * 2),
            kv[:, 0 : self.tp_k_head_num_, :],
            kv[:, self.tp_k_head_num_ : self.tp_k_head_num_ + self.tp_v_head_num_, :],
            o_tensor.view(-1, self.tp_q_head_num_, self.head_dim_ * 2),
            infer_state.position_ids[0],  # [0,0,1,2,3,3,3,4]
            infer_state.b_req_idx,
            infer_state.b_q_start_loc,
            infer_state.b_seq_len,
            infer_state.b_ready_cache_len,
            infer_state.max_q_seq_len,
            infer_state.req_manager.req_to_token_indexs,
            infer_state.b_image_token_tag,
        )
        o3 = o_tensor.view(-1, self.tp_q_head_num_, self.head_dim_ * 2)
        o3 = o3[:, :, : self.head_dim_].contiguous()
        return o3.view(o3.shape[0], -1)

    def _token_attention_kernel(
        self,
        q: torch.Tensor,
        infer_state: NeoChatInferStateInfo,
        layer_weight: NeoChatMOETransformerLayerWeight,
    ) -> torch.Tensor:
        _k, _v = infer_state.mem_manager.get_att_input_params(layer_index=self.layer_num_)
        _q = q.view(-1, self.tp_q_head_num_, self.head_dim_ * 2)
        att_control = AttControl()
        if att_control.scale is None:
            att_control.scale = 1.0 / (self.head_dim_ ** 0.5)
        # att_control.mla_decode_dict["softmax_scale"] = 1.0 / (self.head_dim_ ** 0.5)
        o_tensor = infer_state.decode_att_state.decode_att(
            q=_q, k=_k, v=_v, att_control=att_control, alloc_func=self.alloc_tensor
        )
        o_tensor = o_tensor.view(-1, self.tp_q_head_num_, self.head_dim_ * 2)[:, :, : self.head_dim_].contiguous()
        return o_tensor

    # def _token_attention_kernel(self, q, infer_state: NeoChatInferStateInfo, layer_weight):
    #     total_token_num = infer_state.total_token_num
    #     batch_size = infer_state.batch_size

    #     q_3d = q.view(batch_size, self.tp_q_head_num_, self.head_dim_ * 2)

    #     att_m_tensor = self.alloc_tensor((self.tp_q_head_num_, total_token_num), torch.float32)

    #     k_3d = infer_state.mem_manager.kv_buffer[self.layer_num_][:, 0 : self.tp_k_head_num_, :]
    #     token_att_fwd(
    #         q_3d,
    #         k_3d,
    #         att_m_tensor,
    #         infer_state.req_manager.req_to_token_indexs,
    #         infer_state.b_req_idx,
    #         infer_state.b_kv_start_loc,
    #         infer_state.b_seq_len,
    #         infer_state.max_kv_seq_len,
    #     )

    #     from lightllm.common.basemodel.triton_kernel.att.decode_att.mha.stage3_decode_att import (
    #         token_attention_softmax_and_reducev,
    #     )

    #     token_softmax_reducev_fwd = token_attention_softmax_and_reducev.token_softmax_reducev_fwd

    #     v_3d = infer_state.mem_manager.kv_buffer[self.layer_num_][
    #         :, self.tp_k_head_num_ : self.tp_k_head_num_ + self.tp_v_head_num_, : self.head_dim_
    #     ]

    #     o_3d = self.alloc_tensor((batch_size, self.tp_q_head_num_, self.head_dim_), q.dtype)

    #     token_softmax_reducev_fwd(
    #         att_m_tensor,
    #         v_3d,
    #         o_3d,
    #         infer_state.req_manager.req_to_token_indexs,
    #         infer_state.b_req_idx,
    #         infer_state.b_kv_start_loc,
    #         infer_state.b_seq_len,
    #     )
    #     return o_3d.view(batch_size, -1)
