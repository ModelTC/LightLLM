import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np
from lightllm.models.deepseek2.layer_weights.transformer_layer_weight import Deepseek2TransformerLayerWeight
from lightllm.models.deepseek2.triton_kernel.destindex_copy_kv import destindex_copy_kv
from lightllm.models.deepseek2.triton_kernel.context_flashattention_nopad import (
    context_attention_fwd,
    context_attention_fwd_no_prompt_cache,
)
from lightllm.models.deepseek2.triton_kernel.flash_decoding import token_decode_attention_flash_decoding
from lightllm.models.deepseek2.layer_infer.fused_moe import fused_experts, grouped_topk
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.models.llama.triton_kernel.rmsnorm import rmsnorm_forward
from lightllm.models.chatglm2.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from functools import partial
from lightllm.models.llama.yarn_rotary_utils import get_deepseek_mscale


class Deepseek2TransformerLayerInfer(LlamaTransformerLayerInfer):
    def __init__(self, layer_num, tp_rank, world_size, network_config, mode=[]):
        self.tp_k_head_num_ = 1
        self.tp_v_head_num_ = 1
        self.qk_nope_head_dim = network_config["qk_nope_head_dim"]
        self.qk_rope_head_dim = network_config["qk_rope_head_dim"]
        self.q_lora_rank = network_config["q_lora_rank"]
        self.kv_lora_rank = network_config["kv_lora_rank"]
        self.is_moe = (
            network_config["n_routed_experts"] is not None
            and layer_num >= network_config["first_k_dense_replace"]
            and layer_num % network_config["moe_layer_freq"] == 0
        )

        self.n_shared_experts = network_config["n_shared_experts"]
        self.num_experts_per_tok = network_config["num_experts_per_tok"]
        self.norm_topk_prob = network_config["norm_topk_prob"]
        self.n_group = network_config["n_group"]
        self.topk_group = network_config["topk_group"]
        self.routed_scaling_factor = network_config["routed_scaling_factor"]

        self.softmax_scale = (self.qk_nope_head_dim + self.qk_rope_head_dim) ** (-0.5)
        if network_config.get("rope_scaling", None) is not None:
            self.rope_scaling = network_config["rope_scaling"]
            mscale_all_dim = self.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = self.rope_scaling["factor"]
            if mscale_all_dim:
                mscale = get_deepseek_mscale(scaling_factor, mscale_all_dim)
                self.softmax_scale = self.softmax_scale * mscale * mscale
        super().__init__(layer_num, tp_rank, world_size, network_config, mode)
        self.tp_o_head_num_ = self.tp_q_head_num_
        return

    def _bind_attention(self):
        self._context_attention_kernel = partial(Deepseek2TransformerLayerInfer._context_attention_kernel, self)
        self._token_attention_kernel = partial(
            Deepseek2TransformerLayerInfer._token_decode_attention_flashdecoding, self
        )
        self._copy_kv_to_mem_cache = partial(Deepseek2TransformerLayerInfer._copy_kv_to_mem_cache_normal, self)
        if self.is_moe:
            self._ffn = partial(Deepseek2TransformerLayerInfer._moe_ffn, self)
        else:
            self._ffn = partial(LlamaTransformerLayerInfer._ffn, self)

    def _get_qkv(
        self, input, cache_kv, infer_state: LlamaInferStateInfo, layer_weight: Deepseek2TransformerLayerWeight
    ) -> torch.Tensor:
        if self.q_lora_rank is None:
            q_nope = torch.mm(input.view(-1, self.embed_dim_), layer_weight.fuse_qk_weight_)
            q_rope = torch.mm(input.view(-1, self.embed_dim_), layer_weight.q_rope_proj_)
        else:
            q = torch.mm(input.view(-1, self.embed_dim_), layer_weight.q_a_proj_)
            q = rmsnorm_forward(q, weight=layer_weight.q_a_layernorm_, eps=self.eps_)
            q_nope = torch.mm(q, layer_weight.fuse_qk_weight_)
            q_rope = torch.mm(q, layer_weight.q_rope_proj_)

        q_nope = q_nope.view(-1, self.tp_q_head_num_, self.kv_lora_rank)
        q_rope = q_rope.view(-1, self.tp_q_head_num_, self.qk_rope_head_dim)

        torch.mm(
            input.view(-1, self.embed_dim_),
            layer_weight.kv_a_proj_with_mqa_,
            out=cache_kv.view(-1, self.kv_lora_rank + self.qk_rope_head_dim),
        )
        cache_kv[:, :, : self.kv_lora_rank] = rmsnorm_forward(
            cache_kv[:, :, : self.kv_lora_rank].to(torch.float16), weight=layer_weight.kv_a_layernorm_, eps=self.eps_
        )
        rotary_emb_fwd(
            q_rope,
            cache_kv[:, :, self.kv_lora_rank :],
            infer_state.position_cos,
            infer_state.position_sin,
        )
        return (q_nope, q_rope), cache_kv

    def _get_o(
        self, input, infer_state: LlamaInferStateInfo, layer_weight: Deepseek2TransformerLayerWeight
    ) -> torch.Tensor:
        o_tensor = torch.mm(input.view(-1, self.tp_q_head_num_ * self.kv_lora_rank), layer_weight.fuse_vo_weight_)
        return o_tensor

    def _context_attention_kernel(
        self, q, kv, infer_state: LlamaInferStateInfo, layer_weight, out=None
    ) -> torch.Tensor:
        q_nope, q_rope = q
        o_tensor = torch.empty_like(q_nope) if out is None else out
        if infer_state.use_dynamic_prompt_cache:
            kv = infer_state.mem_manager.kv_buffer[self.layer_num_]
            context_attention_fwd(
                q_nope,
                q_rope,
                kv[:, :, : self.kv_lora_rank],
                kv[:, :, self.kv_lora_rank :],
                o_tensor.view(-1, self.tp_q_head_num_, self.kv_lora_rank),
                infer_state.b_req_idx,
                infer_state.b_start_loc,
                infer_state.b_seq_len,
                infer_state.b_ready_cache_len,
                infer_state.max_len_in_batch,
                infer_state.req_manager.req_to_token_indexs,
                self.softmax_scale,
            )
        else:
            context_attention_fwd_no_prompt_cache(
                q_nope,
                q_rope,
                kv[:, :, : self.kv_lora_rank],
                kv[:, :, self.kv_lora_rank :],
                o_tensor.view(-1, self.tp_q_head_num_, self.kv_lora_rank),
                infer_state.b_start_loc,
                infer_state.b_seq_len,
                infer_state.max_len_in_batch,
                self.softmax_scale,
            )
        q_nope = None
        q_rope = None
        return o_tensor

    def _token_decode_attention_flashdecoding(self, q, infer_state: LlamaInferStateInfo, layer_weight, out=None):
        q_nope, q_rope = q
        kv = infer_state.mem_manager.kv_buffer[self.layer_num_][:, :, : self.kv_lora_rank]
        kv_rope = infer_state.mem_manager.kv_buffer[self.layer_num_][:, :, self.kv_lora_rank :]
        return token_decode_attention_flash_decoding(
            q_nope,
            q_rope,
            kv,
            kv_rope,
            infer_state,
            self.tp_q_head_num_,
            self.kv_lora_rank,
            self.qk_rope_head_dim,
            self.qk_nope_head_dim,
            self.softmax_scale,
            alloc_tensor_func=self.alloc_tensor,
        )

    def _copy_kv_to_mem_cache_normal(self, buffer, mem_index, mem_manager):
        destindex_copy_kv(
            buffer[:, :, : self.kv_lora_rank],
            buffer[:, :, self.kv_lora_rank :],
            mem_index,
            mem_manager.kv_buffer[self.layer_num_][:, :, : self.kv_lora_rank],
            mem_manager.kv_buffer[self.layer_num_][:, :, self.kv_lora_rank :],
        )
        return

    def _moe_ffn(
        self, input, infer_state: LlamaInferStateInfo, layer_weight: Deepseek2TransformerLayerWeight
    ) -> torch.Tensor:
        hidden_states = input.view(-1, self.embed_dim_)
        num_tokens, hidden_dim = hidden_states.shape

        if self.n_shared_experts is not None:
            shared_output = LlamaTransformerLayerInfer._ffn(self, hidden_states, infer_state, layer_weight)

        router_logits = torch.mm(input.view(-1, self.embed_dim_), layer_weight.moe_gate)
        topk_weights, topk_ids = grouped_topk(
            hidden_states,
            router_logits,
            self.num_experts_per_tok,
            renormalize=self.norm_topk_prob,
            num_expert_group=self.n_group,
            topk_group=self.topk_group,
        )

        final_hidden_states = (
            fused_experts(
                hidden_states,
                layer_weight.w1,
                layer_weight.w2,
                topk_weights,
                topk_ids,
                inplace=True,
                alloc_tensor_func=self.alloc_tensor,
            )
            * self.routed_scaling_factor
        )

        if self.n_shared_experts is not None:
            final_hidden_states = final_hidden_states + shared_output
        return final_hidden_states.view(num_tokens, hidden_dim)
