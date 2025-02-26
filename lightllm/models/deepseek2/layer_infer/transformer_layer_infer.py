from typing import Tuple
import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np
from lightllm.models.deepseek2.layer_weights.transformer_layer_weight import Deepseek2TransformerLayerWeight
from lightllm.models.deepseek2.triton_kernel.destindex_copy_kv import destindex_copy_kv
from lightllm.models.deepseek2.triton_kernel.destindex_copy_kv_fp8 import destindex_copy_kv_fp8
from lightllm.models.deepseek2.triton_kernel.context_flashattention_nopad import (
    context_attention_fwd,
    context_attention_fwd_no_prompt_cache,
)
from lightllm.models.deepseek2.triton_kernel.context_flashattention_nopad_fp8 import context_attention_fwd_fp8
from lightllm.models.deepseek2.triton_kernel.context_flashattention_nopad_with_v import context_attention_fwd_with_v
from lightllm.models.deepseek2.triton_kernel.sample_kv import sample_kv

from lightllm.models.deepseek2.triton_kernel.gqa_flash_decoding import gqa_token_decode_attention_flash_decoding
from lightllm.models.deepseek2.triton_kernel.gqa_flash_decoding_fp8 import gqa_token_decode_attention_flash_decoding_fp8
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.models.llama.triton_kernel.rmsnorm import rmsnorm_forward
from lightllm.models.llama.triton_kernel.silu_and_mul import silu_and_mul_fwd
from lightllm.models.deepseek2.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.models.deepseek2.infer_struct import Deepseek2InferStateInfo
from lightllm.models.deepseek2.flashinfer_struct import Deepseek2FlashInferStateInfo
from functools import partial
from lightllm.models.llama.yarn_rotary_utils import get_deepseek_mscale
import os
from lightllm.utils.envs_utils import enable_env_vars


class Deepseek2TransformerLayerInfer(LlamaTransformerLayerInfer):
    def __init__(self, layer_num, tp_rank, world_size, network_config, mode=[]):
        self.tp_k_head_num_ = 1
        self.tp_v_head_num_ = 1
        self.qk_nope_head_dim = network_config["qk_nope_head_dim"]
        self.qk_rope_head_dim = network_config["qk_rope_head_dim"]
        self.v_head_dim = network_config["v_head_dim"]
        self.q_lora_rank = network_config["q_lora_rank"]
        self.kv_lora_rank = network_config["kv_lora_rank"]

        self.n_routed_experts = network_config["n_routed_experts"]

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
        self.enable_cc_method = not os.getenv("DISABLE_CC_METHOD", "False").upper() in ["ON", "TRUE", "1"]
        super().__init__(layer_num, tp_rank, world_size, network_config, mode)
        self.enable_dp = os.getenv("ENABLE_DP", "0").upper() in ["ON", "TRUE", "1"]
        if self.enable_dp:
            self.tp_q_head_num_ = int(self.tp_q_head_num_ * self.world_size_)
        self.tp_o_head_num_ = self.tp_q_head_num_
        self.num_heads = network_config["num_attention_heads"]
        self.num_kv_heads = network_config["num_key_value_heads"]
        return

    def _bind_func(self):
        super()._bind_func()
        self._bind_ffn()
        return

    def _bind_ffn(self):
        if self.is_moe:
            if self.enable_dp:
                if os.environ.get("MOE_MODE", "TP") == "TP":
                    self._ffn = partial(Deepseek2TransformerLayerInfer._moe_ffn_dtp, self)
            else:
                if os.environ.get("ETP_MODE_ENABLED") == "true":
                    self._ffn = partial(Deepseek2TransformerLayerInfer._moe_ffn_etp, self)
                else:
                    self._ffn = partial(Deepseek2TransformerLayerInfer._moe_ffn, self)
        else:
            self._ffn = partial(LlamaTransformerLayerInfer._ffn, self)

    def _bind_attention(self):
        if "triton_fp8kv" in self.mode:
            self._copy_kv_to_mem_cache = partial(Deepseek2TransformerLayerInfer._copy_kv_to_mem_cache_fp8, self)
            self._token_attention_kernel = partial(
                Deepseek2TransformerLayerInfer._token_gqa_decode_attention_flashdecoding_fp8, self
            )
        else:
            self._copy_kv_to_mem_cache = partial(Deepseek2TransformerLayerInfer._copy_kv_to_mem_cache_normal, self)
            if enable_env_vars("ENABLE_FLASHINFER_DECODE_MLA"):
                self._token_attention_kernel = partial(
                    Deepseek2TransformerLayerInfer._token_gqa_decode_attention_flashinfer, self
                )
            else:
                self._token_attention_kernel = partial(
                    Deepseek2TransformerLayerInfer._token_gqa_decode_attention_flashdecoding, self
                )
        if self.enable_cc_method:
            if "triton_fp8kv" in self.mode:
                if enable_env_vars("ENABLE_FLASHINFER_PREFILLED"):
                    self._context_attention_kernel = partial(
                        Deepseek2TransformerLayerInfer._context_attention_flashinfer_kernel_with_CC_fp8, self
                    )
                else:
                    self._context_attention_kernel = partial(
                        Deepseek2TransformerLayerInfer._context_attention_kernel_with_CC_fp8, self
                    )
            else:
                if enable_env_vars("ENABLE_FLASHINFER_PREFILLED"):
                    self._context_attention_kernel = partial(
                        Deepseek2TransformerLayerInfer._context_attention_flashinfer_kernel_with_CC, self
                    )
                else:
                    self._context_attention_kernel = partial(
                        Deepseek2TransformerLayerInfer._context_attention_kernel_with_CC, self
                    )
        else:
            if "triton_fp8kv" in self.mode:
                self._context_attention_kernel = partial(
                    Deepseek2TransformerLayerInfer._context_attention_kernel_origin_fp8, self
                )
            else:
                self._context_attention_kernel = partial(
                    Deepseek2TransformerLayerInfer._context_attention_kernel_origin, self
                )

    def _get_qkv(
        self,
        input: torch.Tensor,
        cache_kv,
        infer_state: Deepseek2InferStateInfo,
        layer_weight: Deepseek2TransformerLayerWeight,
    ) -> torch.Tensor:
        input = input.view(-1, self.embed_dim_)

        if self.q_lora_rank is None:
            q = layer_weight.q_weight_.mm(input)
        else:
            q = layer_weight.q_a_proj_.mm(input)
            rmsnorm_forward(q, weight=layer_weight.q_a_layernorm_.weight, eps=self.eps_, out=q)
            q = layer_weight.q_b_proj_.mm(q)
        q = q.view(-1, self.tp_q_head_num_, self.qk_nope_head_dim + self.qk_rope_head_dim)
        q_nope, q_rope = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        layer_weight.kv_a_proj_with_mqa_.mm(input, out=cache_kv.view(-1, self.kv_lora_rank + self.qk_rope_head_dim))
        rmsnorm_forward(
            cache_kv[:, :, : self.kv_lora_rank],
            weight=layer_weight.kv_a_layernorm_.weight,
            eps=self.eps_,
            out=cache_kv[:, :, : self.kv_lora_rank],
        )

        rotary_emb_fwd(
            q_rope,
            cache_kv[:, :, self.kv_lora_rank :],
            infer_state.position_cos,
            infer_state.position_sin,
        )
        return q, cache_kv

    def _get_o(
        self, input: torch.Tensor, infer_state: Deepseek2InferStateInfo, layer_weight: Deepseek2TransformerLayerWeight
    ) -> torch.Tensor:
        if input.shape[2] == self.kv_lora_rank:
            input = layer_weight.v_b_proj_.bmm(input.transpose(0, 1)).transpose(0, 1)
        o_tensor = layer_weight.o_weight_.mm(input.reshape(-1, self.tp_q_head_num_ * self.qk_nope_head_dim))
        return o_tensor

    def _decompress_kv(
        self, kv, infer_state: Deepseek2InferStateInfo, layer_weight: Deepseek2TransformerLayerWeight, is_fp8
    ):
        if infer_state.use_dynamic_prompt_cache:
            if is_fp8:
                kv = infer_state.mem_manager.kv_buffer[self.layer_num_][:, :, :-2].view(torch.float8_e4m3fn)
                kv_scale = infer_state.mem_manager.kv_buffer[self.layer_num_][:, :, -2:].view(torch.bfloat16)
                k_scale = self.alloc_tensor([infer_state.total_token_num, 1], dtype=kv_scale.dtype)
            else:
                kv = infer_state.mem_manager.kv_buffer[self.layer_num_]
                kv_scale = None
                k_scale = None

            compressed_kv = self.alloc_tensor(
                [infer_state.total_token_num, 1, layer_weight.kv_lora_rank], dtype=kv.dtype
            )
            k_rope = self.alloc_tensor([infer_state.total_token_num, 1, self.qk_rope_head_dim], dtype=kv.dtype)
            sample_kv(
                kv,
                compressed_kv,
                k_rope,
                infer_state.b_req_idx,
                infer_state.b_seq_len,
                infer_state.req_manager.req_to_token_indexs,
                infer_state.b_kv_start_loc,
                kv_scale,
                k_scale,
            )
            if k_scale is not None:
                compressed_kv = compressed_kv.to(k_scale.dtype) * k_scale.unsqueeze(-1)
                k_rope = k_rope.to(k_scale.dtype) * k_scale.unsqueeze(-1)
        else:
            compressed_kv, k_rope = torch.split(  # (b*s, 1, kv_lora + qk_r)
                kv, [layer_weight.kv_lora_rank, layer_weight.qk_rope_head_dim], dim=-1
            )

        # CC
        compressed_kv = compressed_kv.view(-1, layer_weight.kv_lora_rank).contiguous()
        kv_nope = self.alloc_tensor(
            [compressed_kv.shape[0], self.tp_q_head_num_, (self.qk_nope_head_dim + self.v_head_dim)],
            dtype=compressed_kv.dtype,
        )
        layer_weight.cc_kv_b_proj_.mm(compressed_kv, out=kv_nope.reshape(compressed_kv.shape[0], -1))
        k_nope, v = torch.split(kv_nope, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        return k_nope, k_rope, v

    def _context_attention_flashinfer_kernel_with_CC(
        self,
        q: torch.Tensor,
        kv,
        infer_state: Deepseek2FlashInferStateInfo,
        layer_weight: Deepseek2TransformerLayerWeight,
        out=None,
    ) -> torch.Tensor:
        k_nope, k_rope, v = self._decompress_kv(kv, infer_state, layer_weight, False)
        o_tensor = (
            self.alloc_tensor((q.shape[0], q.shape[1], self.qk_nope_head_dim), dtype=q.dtype) if out is None else out
        )
        k = torch.cat([k_nope, torch.repeat_interleave(k_rope, self.tp_q_head_num_, dim=-2)], dim=-1)
        infer_state.prefill_wrapper.run(q, k, v, out=o_tensor)
        return o_tensor

    def _context_attention_flashinfer_kernel_with_CC_fp8(
        self,
        q: torch.Tensor,
        kv,
        infer_state: Deepseek2FlashInferStateInfo,
        layer_weight: Deepseek2TransformerLayerWeight,
        out=None,
    ) -> torch.Tensor:
        k_nope, k_rope, v = self._decompress_kv(kv, infer_state, layer_weight, True)
        o_tensor = (
            self.alloc_tensor((q.shape[0], q.shape[1], self.qk_nope_head_dim), dtype=q.dtype) if out is None else out
        )
        k = torch.cat([k_nope, torch.repeat_interleave(k_rope, self.tp_q_head_num_, dim=-2)], dim=-1)
        infer_state.prefill_wrapper.run(q, k, v, out=o_tensor)
        return o_tensor

    def _context_attention_kernel_with_CC(
        self,
        q: torch.Tensor,
        kv,
        infer_state: Deepseek2InferStateInfo,
        layer_weight: Deepseek2TransformerLayerWeight,
        out=None,
    ) -> torch.Tensor:
        k_nope, k_rope, v = self._decompress_kv(kv, infer_state, layer_weight, False)
        q_nope, q_rope = q[:, :, : -self.qk_rope_head_dim], q[:, :, -self.qk_rope_head_dim :]
        o_tensor = self.alloc_tensor(q_nope.shape, dtype=q_nope.dtype) if out is None else out
        context_attention_fwd_with_v(
            q_nope,
            q_rope,
            k_nope,
            k_rope,
            v,
            o_tensor.view(-1, self.tp_q_head_num_, q_nope.shape[-1]),
            infer_state.b_start_loc,
            infer_state.b_kv_start_loc,
            infer_state.b_seq_len,
            infer_state.b_ready_cache_len,
            infer_state.max_len_in_batch,
            self.softmax_scale,
        )
        return o_tensor

    def _context_attention_kernel_with_CC_fp8(
        self,
        q: torch.Tensor,
        kv,
        infer_state: Deepseek2InferStateInfo,
        layer_weight: Deepseek2TransformerLayerWeight,
        out=None,
    ) -> torch.Tensor:
        k_nope, k_rope, v = self._decompress_kv(kv, infer_state, layer_weight, True)
        q_nope, q_rope = q[:, :, : -self.qk_rope_head_dim], q[:, :, -self.qk_rope_head_dim :]
        o_tensor = self.alloc_tensor(q_nope.shape, dtype=q_nope.dtype) if out is None else out
        context_attention_fwd_with_v(
            q_nope,
            q_rope,
            k_nope,
            k_rope,
            v,
            o_tensor.view(-1, self.tp_q_head_num_, q_nope.shape[-1]),
            infer_state.b_start_loc,
            infer_state.b_kv_start_loc,
            infer_state.b_seq_len,
            infer_state.b_ready_cache_len,
            infer_state.max_len_in_batch,
            self.softmax_scale,
        )
        return o_tensor

    def _context_attention_kernel_origin(
        self,
        q: torch.Tensor,
        kv,
        infer_state: Deepseek2InferStateInfo,
        layer_weight: Deepseek2TransformerLayerWeight,
        out=None,
    ) -> torch.Tensor:
        q_nope, q_rope = q[:, :, : -self.qk_rope_head_dim], q[:, :, -self.qk_rope_head_dim :]
        q_nope = layer_weight.k_b_proj_.bmm(q_nope.transpose(0, 1)).transpose(0, 1)
        o_tensor = self.alloc_tensor(q_nope.shape, dtype=q_nope.dtype) if out is None else out
        if infer_state.use_dynamic_prompt_cache:
            kv = infer_state.mem_manager.kv_buffer[self.layer_num_]
            context_attention_fwd(
                q_nope,
                q_rope,
                kv[:, :, : -self.qk_rope_head_dim],
                kv[:, :, -self.qk_rope_head_dim :],
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
                kv[:, :, : -self.qk_rope_head_dim],
                kv[:, :, -self.qk_rope_head_dim :],
                o_tensor.view(-1, self.tp_q_head_num_, self.kv_lora_rank),
                infer_state.b_start_loc,
                infer_state.b_seq_len,
                infer_state.max_len_in_batch,
                self.softmax_scale,
            )

        return o_tensor

    def _context_attention_kernel_origin_fp8(
        self,
        q: torch.Tensor,
        kv,
        infer_state: Deepseek2InferStateInfo,
        layer_weight: Deepseek2TransformerLayerWeight,
        out=None,
    ) -> torch.Tensor:
        q_nope, q_rope = q[:, :, : -self.qk_rope_head_dim], q[:, :, -self.qk_rope_head_dim :]
        q_nope = layer_weight.k_b_proj_.bmm(q_nope.transpose(0, 1)).transpose(0, 1)
        o_tensor = self.alloc_tensor(q_nope.shape, dtype=q_nope.dtype) if out is None else out
        if infer_state.use_dynamic_prompt_cache:
            kv = infer_state.mem_manager.kv_buffer[self.layer_num_][:, :, :-2].view(torch.float8_e4m3fn)
            kv_scale = infer_state.mem_manager.kv_buffer[self.layer_num_][:, :, -2:].view(torch.bfloat16)
            context_attention_fwd_fp8(
                q_nope,
                q_rope,
                kv[:, :, : -self.qk_rope_head_dim],
                kv[:, :, -self.qk_rope_head_dim :],
                kv_scale,
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
                kv[:, :, : -self.qk_rope_head_dim],
                kv[:, :, -self.qk_rope_head_dim :],
                o_tensor.view(-1, self.tp_q_head_num_, self.kv_lora_rank),
                infer_state.b_start_loc,
                infer_state.b_seq_len,
                infer_state.max_len_in_batch,
                self.softmax_scale,
            )

        return o_tensor

    def _token_gqa_decode_attention_flashinfer(
        self, q, infer_state: Deepseek2FlashInferStateInfo, layer_weight: Deepseek2TransformerLayerWeight, out=None
    ):
        q_nope, q_rope = q[:, :, : -self.qk_rope_head_dim], q[:, :, -self.qk_rope_head_dim :]
        q_nope = layer_weight.k_b_proj_.bmm(q_nope.transpose(0, 1)).transpose(0, 1)

        kv = infer_state.mem_manager.kv_buffer[self.layer_num_]
        o_tensor = self.alloc_tensor(q_nope.shape, dtype=q_nope.dtype)

        infer_state.decode_wrapper.run(
            q_nope,
            q_rope,
            kv[:, :, : -self.qk_rope_head_dim],
            kv[:, :, -self.qk_rope_head_dim :],
            out=o_tensor,
            return_lse=False,
        )
        return o_tensor

    def _token_gqa_decode_attention_flashdecoding(
        self, q, infer_state: Deepseek2InferStateInfo, layer_weight: Deepseek2TransformerLayerWeight, out=None
    ):
        q_nope, q_rope = q[:, :, : -self.qk_rope_head_dim], q[:, :, -self.qk_rope_head_dim :]
        q_nope = layer_weight.k_b_proj_.bmm(q_nope.transpose(0, 1)).transpose(0, 1)

        kv = infer_state.mem_manager.kv_buffer[self.layer_num_]
        o_tensor = self.alloc_tensor(q_nope.shape, dtype=q_nope.dtype)

        if enable_env_vars("ENABLE_OPT_DECODE_MHA"):
            q = torch.cat([q_nope, q_rope], dim=-1)
            q_nope, q_rope = None, None
            import lightllm_ppl_mla

            lightllm_ppl_mla.decode_mla(
                o_tensor,
                q,
                kv,
                infer_state.req_manager.req_to_token_indexs,
                infer_state.kv_starts,
                infer_state.b_req_idx,
                self.softmax_scale,
                q.shape[-1],
                self.kv_lora_rank,
            )
            return o_tensor
        else:
            kv = infer_state.mem_manager.kv_buffer[self.layer_num_]
            return gqa_token_decode_attention_flash_decoding(
                q_nope,
                q_rope,
                kv[:, :, : -self.qk_rope_head_dim],
                kv[:, :, -self.qk_rope_head_dim :],
                infer_state,
                self.tp_q_head_num_,
                self.kv_lora_rank,
                self.qk_rope_head_dim,
                self.qk_nope_head_dim,
                self.softmax_scale,
                alloc_tensor_func=self.alloc_tensor,
            )

    def _token_gqa_decode_attention_flashdecoding_fp8(
        self, q, infer_state: Deepseek2InferStateInfo, layer_weight: Deepseek2TransformerLayerWeight, out=None
    ):
        q_nope, q_rope = q[:, :, : -self.qk_rope_head_dim], q[:, :, -self.qk_rope_head_dim :]
        q_nope = layer_weight.k_b_proj_.bmm(q_nope.transpose(0, 1)).transpose(0, 1)

        kv = infer_state.mem_manager.kv_buffer[self.layer_num_][:, :, :-2].view(torch.float8_e4m3fn)
        kv_scale = infer_state.mem_manager.kv_buffer[self.layer_num_][:, :, -2:].view(torch.bfloat16)
        return gqa_token_decode_attention_flash_decoding_fp8(
            q_nope,
            q_rope,
            kv[:, :, : -self.qk_rope_head_dim],
            kv[:, :, -self.qk_rope_head_dim :],
            kv_scale,
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

    def _copy_kv_to_mem_cache_fp8(self, buffer, mem_index, mem_manager):
        destindex_copy_kv_fp8(
            buffer[:, :, : self.kv_lora_rank],
            buffer[:, :, self.kv_lora_rank :],
            mem_index,
            mem_manager.kv_buffer[self.layer_num_][:, :, : self.kv_lora_rank].view(torch.float8_e4m3fn),
            mem_manager.kv_buffer[self.layer_num_][:, :, self.kv_lora_rank : -2].view(torch.float8_e4m3fn),
            mem_manager.kv_buffer[self.layer_num_][:, :, -2:].view(buffer.dtype),
        )
        return

    def _ffn_dp(
        self, input, infer_state: Deepseek2InferStateInfo, layer_weight: Deepseek2TransformerLayerWeight
    ) -> torch.Tensor:
        tp_hidden_states = input.view(-1, self.embed_dim_)
        num_tokens, hidden_dim = tp_hidden_states.shape
        hidden_states = self.alloc_tensor(
            [infer_state.all_token_num, hidden_dim], dtype=tp_hidden_states.dtype, device=tp_hidden_states.device
        )
        dist.all_gather_into_tensor(
            hidden_states,
            tp_hidden_states,
            group=None,
            async_op=False,
        )
        up_gate_out = layer_weight.gate_up_proj.mm(hidden_states)
        ffn1_out = self.alloc_tensor((hidden_states.size(0), up_gate_out.size(1) // 2), input.dtype)
        silu_and_mul_fwd(up_gate_out, ffn1_out)
        input = None
        up_gate_out = None
        ffn2_out = layer_weight.down_proj.mm(ffn1_out)
        if self.world_size_ > 1:
            dist.all_reduce(ffn2_out, op=dist.ReduceOp.SUM, async_op=False)
        ffn1_out = None
        return ffn2_out[infer_state.start_idx : infer_state.end_idx]

    def _moe_ffn(
        self, input, infer_state: Deepseek2InferStateInfo, layer_weight: Deepseek2TransformerLayerWeight
    ) -> torch.Tensor:

        hidden_states = input.view(-1, self.embed_dim_)
        num_tokens, hidden_dim = hidden_states.shape

        if self.n_shared_experts is not None:
            shared_output = LlamaTransformerLayerInfer._ffn(self, hidden_states, infer_state, layer_weight)

        router_logits = layer_weight.moe_gate.mm(hidden_states)
        layer_weight.experts.experts(
            hidden_states,
            router_logits=router_logits,
            top_k=self.num_experts_per_tok,
            renormalize=self.norm_topk_prob,
            use_grouped_topk=self.n_group,
            topk_group=self.topk_group,
            num_expert_group=self.n_group,
        )

        hidden_states.mul_(self.routed_scaling_factor)

        if self.n_shared_experts is not None:
            hidden_states.add_(shared_output)

        return hidden_states.view(num_tokens, hidden_dim)

    def _moe_ffn_dtp(
        self, input, infer_state: Deepseek2InferStateInfo, layer_weight: Deepseek2TransformerLayerWeight
    ) -> torch.Tensor:

        tp_hidden_states = input.view(-1, self.embed_dim_)
        num_tokens, hidden_dim = tp_hidden_states.shape
        hidden_states = self.alloc_tensor(
            [infer_state.all_token_num, hidden_dim], dtype=tp_hidden_states.dtype, device=tp_hidden_states.device
        )
        if infer_state.is_prefill:
            dist.all_gather(
                [
                    hidden_states[infer_state.all_start_idx[i] : infer_state.all_end_idx[i], :]
                    for i in range(self.world_size_)
                ],
                tp_hidden_states,
                group=None,
                async_op=False,
            )
        else:
            dist.all_gather_into_tensor(
                hidden_states,
                tp_hidden_states,
                group=None,
                async_op=False,
            )
        if self.n_shared_experts is not None:
            shared_output = LlamaTransformerLayerInfer._ffn(self, hidden_states, infer_state, layer_weight)

        router_logits = layer_weight.moe_gate.mm(hidden_states)
        layer_weight.experts.experts(
            hidden_states,
            router_logits=router_logits,
            top_k=self.num_experts_per_tok,
            renormalize=self.norm_topk_prob,
            use_grouped_topk=self.n_group,
            topk_group=self.topk_group,
            num_expert_group=self.n_group,
        )

        hidden_states.mul_(self.routed_scaling_factor)

        if self.n_shared_experts is not None:
            hidden_states.add_(shared_output)

        hidden_states = hidden_states.view(infer_state.all_token_num, hidden_dim)
        if self.world_size_ > 1:
            dist.all_reduce(hidden_states, op=dist.ReduceOp.SUM, async_op=False)
        return hidden_states[infer_state.start_idx : infer_state.end_idx]

    def _moe_ffn_etp(
        self, input, infer_state: Deepseek2InferStateInfo, layer_weight: Deepseek2TransformerLayerWeight
    ) -> torch.Tensor:
        world_size_ = self.world_size_
        # num_local_experts = self.n_shared_experts // world_size_
        # local_expert_offset = self.tp_rank_ * num_local_experts
        num_experts_per_token = self.num_experts_per_tok
        num_experts = self.n_routed_experts
        # num_expert_groups = self.n_group
        # num_groups_per_token = self.topk_group
        gating_scaling_factor = self.routed_scaling_factor
        # gating_normalize_prob = self.norm_topk_prob
        rank_self = self.tp_rank_

        hidden_states = input.view(-1, self.embed_dim_)
        num_tokens, hidden_dim = hidden_states.shape

        final_hidden_states = torch.empty(
            num_tokens, hidden_dim, device=hidden_states.device, dtype=hidden_states.dtype
        )

        # router_logits_len = hidden_states.shape[0]*layer_weight.moe_gate.shape[1]
        router_logits = layer_weight.moe_gate.mm(hidden_states)

        # now some parameter is not supported yet
        # assert gating_normalize_prob is False
        # assert num_expert_groups<=1

        import lightllm_moe_etp_kernel

        lightllm_moe_etp_kernel.moe_fused_all(
            router_logits.contiguous(),
            hidden_states.contiguous(),
            layer_weight.gate_up_proj.weight.contiguous(),  # transpose
            layer_weight.down_proj.weight.contiguous(),  # transpose
            layer_weight.experts.expert_gate_up_proj_etp.contiguous(),
            layer_weight.experts.expert_down_proj_etp.contiguous(),
            infer_state.mem_manager.work_buffer.contiguous(),
            infer_state.mem_manager.work_buffer.nelement(),
            final_hidden_states.contiguous(),
            rank_self,
            gating_scaling_factor,
            num_experts,
            num_experts_per_token,
            num_tokens,
            world_size_,
            hidden_dim,
            layer_weight.gate_up_proj.weight.size(1) // 2,
            layer_weight.experts.expert_gate_up_proj_etp.size(1) // 2,
            self.n_shared_experts is not None,
        )

        router_logits = None

        return final_hidden_states.view(num_tokens, hidden_dim)
