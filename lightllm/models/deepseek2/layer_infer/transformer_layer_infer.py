import os
import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np
import triton
from typing import Tuple
from lightllm.models.deepseek2.layer_weights.transformer_layer_weight import Deepseek2TransformerLayerWeight
from lightllm.models.deepseek2.triton_kernel.destindex_copy_kv import destindex_copy_kv
from lightllm.models.deepseek2.triton_kernel.destindex_copy_kv_fp8 import destindex_copy_kv_fp8
from lightllm.models.deepseek2.triton_kernel.context_flashattention_nopad import (
    context_attention_fwd,
)
from lightllm.models.deepseek2.triton_kernel.context_flashattention_nopad_fp8 import context_attention_fwd_fp8
from lightllm.models.deepseek2.triton_kernel.context_flashattention_nopad_with_v import context_attention_fwd_with_v
from lightllm.models.deepseek2.triton_kernel.sample_kv import sample_kv
from lightllm.models.deepseek2.triton_kernel.repeat_rope import repeat_rope
from lightllm.models.deepseek2.triton_kernel.gqa_flash_decoding import gqa_token_decode_attention_flash_decoding
from lightllm.models.deepseek2.triton_kernel.gqa_flash_decoding_fp8 import gqa_token_decode_attention_flash_decoding_fp8
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.models.llama.triton_kernel.rmsnorm import rmsnorm_forward
from lightllm.models.deepseek2.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.models.deepseek2.infer_struct import Deepseek2InferStateInfo
from lightllm.models.deepseek2.flashinfer_struct import Deepseek2FlashInferStateInfo
from lightllm.models.deepseek2.flashattention_infer_struct import Deepseek2FlashAttentionStateInfo
from functools import partial
from lightllm.models.llama.yarn_rotary_utils import get_deepseek_mscale
from lightllm.distributed.communication_op import all_gather, all_gather_into_tensor, all_reduce, reduce_scatter_tensor
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.dist_utils import get_global_world_size
from lightllm.utils.log_utils import init_logger
from lightllm.utils.sgl_utils import flash_attn_varlen_func, flash_attn_with_kvcache, merge_state_v2

logger = init_logger(__name__)


class Deepseek2TransformerLayerInfer(LlamaTransformerLayerInfer):
    def __init__(self, layer_num, network_config, mode=[]):
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

        self.softmax_scale = (self.qk_nope_head_dim + self.qk_rope_head_dim) ** (-0.5)
        if network_config.get("rope_scaling", None) is not None:
            self.rope_scaling = network_config["rope_scaling"]
            mscale_all_dim = self.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = self.rope_scaling["factor"]
            if mscale_all_dim:
                mscale = get_deepseek_mscale(scaling_factor, mscale_all_dim)
                self.softmax_scale = self.softmax_scale * mscale * mscale
        self.enable_cc_method = not os.getenv("DISABLE_CC_METHOD", "False").upper() in ["ON", "TRUE", "1"]
        super().__init__(layer_num, network_config, mode)
        self.num_heads = network_config["num_attention_heads"]
        self.num_kv_heads = network_config["num_key_value_heads"]
        return

    def _bind_func(self):
        super()._bind_func()
        self._bind_ffn()
        return

    def _bind_ffn(self):
        if self.is_moe:
            moe_mode = os.environ.get("MOE_MODE", "TP")
            if moe_mode == "EP":
                self._ffn = partial(Deepseek2TransformerLayerInfer._moe_ffn_edp, self)
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
            if get_env_start_args().enable_fa3:
                self._token_attention_kernel = partial(
                    Deepseek2TransformerLayerInfer._token_gqa_decode_attention_flashattention, self
                )
            elif get_env_start_args().enable_flashinfer_decode:
                self._token_attention_kernel = partial(
                    Deepseek2TransformerLayerInfer._token_gqa_decode_attention_flashinfer, self
                )
            else:
                self._token_attention_kernel = partial(
                    Deepseek2TransformerLayerInfer._token_gqa_decode_attention_flashdecoding, self
                )
        if self.enable_cc_method:
            if "triton_fp8kv" in self.mode:
                if get_env_start_args().enable_flashinfer_prefill:
                    self._context_attention_kernel = partial(
                        Deepseek2TransformerLayerInfer._context_attention_flashinfer_kernel_with_CC_fp8, self
                    )
                else:
                    self._context_attention_kernel = partial(
                        Deepseek2TransformerLayerInfer._context_attention_kernel_with_CC_fp8, self
                    )
            else:
                if get_env_start_args().enable_fa3:
                    self._context_attention_kernel = partial(
                        Deepseek2TransformerLayerInfer._context_attention_flashattention_kernel_with_CC, self
                    )
                elif get_env_start_args().enable_flashinfer_prefill:
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

    def _tpsp_get_qkv(
        self, input, cache_kv, infer_state: Deepseek2InferStateInfo, layer_weight: Deepseek2TransformerLayerWeight
    ) -> torch.Tensor:
        if self.tp_world_size_ > 1:
            sp_token_num, hidden_dim = input.shape
            gather_input = self.alloc_tensor(
                (sp_token_num * self.tp_world_size_, hidden_dim), dtype=input.dtype, device=input.device
            )
            all_gather_into_tensor(gather_input, input, group=infer_state.dist_group, async_op=False)
            input = gather_input[0 : len(infer_state.position_cos), :]

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

    def _tpsp_get_o(
        self, input, infer_state: Deepseek2InferStateInfo, layer_weight: Deepseek2TransformerLayerWeight
    ) -> torch.Tensor:
        if input.shape[2] == self.kv_lora_rank:
            input = layer_weight.v_b_proj_.bmm(input.transpose(0, 1)).transpose(0, 1)

        input = input.reshape(-1, self.tp_q_head_num_ * self.qk_nope_head_dim)
        dest_size = triton.cdiv(input.shape[0], self.tp_world_size_) * self.tp_world_size_
        o_tensor = self.alloc_tensor((dest_size, self.embed_dim_), dtype=input.dtype, device=input.device)
        layer_weight.o_weight_.mm(input, out=o_tensor[0 : len(infer_state.position_cos), :])
        if self.tp_world_size_ > 1:
            sp_token_num = o_tensor.shape[0] // self.tp_world_size_
            reduce_o_tensor = self.alloc_tensor((sp_token_num, self.embed_dim_), dtype=input.dtype, device=input.device)
            reduce_scatter_tensor(
                output=reduce_o_tensor,
                input=o_tensor,
                op=dist.ReduceOp.SUM,
                group=infer_state.dist_group,
                async_op=False,
            )
            o_tensor = reduce_o_tensor

        return o_tensor

    def _decompress_kv(
        self,
        kv,
        infer_state: Deepseek2InferStateInfo,
        layer_weight: Deepseek2TransformerLayerWeight,
        is_fp8,
        total_token_num,
        b_seq_len,
        max_seq_len,
        b_kv_start_loc,
        skip_sample=False,
    ):
        if not skip_sample:
            if is_fp8:
                kv = infer_state.mem_manager.kv_buffer[self.layer_num_][:, :, :-2].view(torch.float8_e4m3fn)
                kv_scale = infer_state.mem_manager.kv_buffer[self.layer_num_][:, :, -2:].view(torch.bfloat16)
                k_scale = self.alloc_tensor([total_token_num, 1], dtype=kv_scale.dtype)
            else:
                kv = infer_state.mem_manager.kv_buffer[self.layer_num_]
                kv_scale = None
                k_scale = None

            compressed_kv = self.alloc_tensor([total_token_num, 1, layer_weight.kv_lora_rank], dtype=kv.dtype)
            k_rope = self.alloc_tensor([total_token_num, 1, self.qk_rope_head_dim], dtype=kv.dtype)
            sample_kv(
                kv,
                compressed_kv,
                k_rope,
                infer_state.b_req_idx,
                max_seq_len,
                b_seq_len,
                infer_state.req_manager.req_to_token_indexs,
                b_kv_start_loc,
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

    # Adapted from:
    # https://github.com/sgl-project/sglang/blob/c998d04b46920f06d945fbef9023884a768723fc/python/sglang/srt/models/deepseek_v2.py#L962
    def _context_attention_flashattention_kernel_with_CC(
        self,
        q: torch.Tensor,
        kv,
        infer_state: Deepseek2FlashAttentionStateInfo,
        layer_weight: Deepseek2TransformerLayerWeight,
        out=None,
    ) -> torch.Tensor:
        k_nope, k_rope, v = self._decompress_kv(
            kv,
            infer_state,
            layer_weight,
            False,
            infer_state.total_token_num,
            infer_state.b_seq_len,
            infer_state.max_value_in_b_seq_len,
            infer_state.b1_kv_start_loc,
            skip_sample=True,
        )
        k = torch.cat([k_nope, torch.repeat_interleave(k_rope, self.tp_q_head_num_, dim=-2)], dim=-1)
        o_tensor, lse, *rest = flash_attn_varlen_func(
            q=q.view(-1, self.tp_q_head_num_, self.qk_nope_head_dim + self.qk_rope_head_dim),
            k=k.view(-1, self.tp_k_head_num_, self.qk_nope_head_dim + self.qk_rope_head_dim),
            v=v.view(-1, self.tp_v_head_num_, self.v_head_dim),
            cu_seqlens_q=infer_state.cu_seqlens_q,
            cu_seqlens_k=infer_state.cu_seqlens_q,
            max_seqlen_q=infer_state.q_max_seq_len,
            max_seqlen_k=infer_state.max_seq_len,
            softmax_scale=self.softmax_scale,
            causal=True,
            return_softmax_lse=True,
        )
        if infer_state.has_prefix_kv:
            k_nope, k_rope, v = self._decompress_kv(
                kv,
                infer_state,
                layer_weight,
                False,
                infer_state.prefix_total_token_num,
                infer_state.b_ready_cache_len,
                infer_state.prefix_k_max_len,
                infer_state.cu_seqlens_prefix_k,
            )
            k = torch.cat([k_nope, torch.repeat_interleave(k_rope, self.tp_q_head_num_, dim=-2)], dim=-1)
            prefix_output, prefix_lse, *rest = flash_attn_varlen_func(
                q=q.view(-1, self.tp_q_head_num_, self.qk_nope_head_dim + self.qk_rope_head_dim),
                k=k.view(-1, self.tp_k_head_num_, self.qk_nope_head_dim + self.qk_rope_head_dim),
                v=v.view(-1, self.tp_v_head_num_, self.v_head_dim),
                cu_seqlens_q=infer_state.cu_seqlens_q,
                cu_seqlens_k=infer_state.cu_seqlens_prefix_k,
                max_seqlen_q=infer_state.q_max_seq_len,
                max_seqlen_k=infer_state.prefix_k_max_len,
                softmax_scale=self.softmax_scale,
                causal=False,
                return_softmax_lse=True,
            )
            lse = torch.transpose(lse, 0, 1).contiguous()
            prefix_lse = torch.transpose(prefix_lse, 0, 1).contiguous()
            tmp_output = (
                self.alloc_tensor((q.shape[0], q.shape[1], self.qk_nope_head_dim), dtype=q.dtype)
                if out is None
                else out
            )
            tmp_lse = torch.empty_like(lse)
            merge_state_v2(prefix_output, prefix_lse, o_tensor, lse, tmp_output, tmp_lse)
            o_tensor = tmp_output
        return o_tensor

    def _context_attention_flashinfer_kernel_with_CC(
        self,
        q: torch.Tensor,
        kv,
        infer_state: Deepseek2FlashInferStateInfo,
        layer_weight: Deepseek2TransformerLayerWeight,
        out=None,
    ) -> torch.Tensor:
        k_nope, k_rope, v = self._decompress_kv(
            kv,
            infer_state,
            layer_weight,
            False,
            infer_state.total_token_num,
            infer_state.b_seq_len,
            infer_state.max_value_in_b_seq_len,
            infer_state.b1_kv_start_loc,
        )
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
        k_nope, k_rope, v = self._decompress_kv(
            kv,
            infer_state,
            layer_weight,
            True,
            infer_state.total_token_num,
            infer_state.b_seq_len,
            infer_state.max_value_in_b_seq_len,
            infer_state.b1_kv_start_loc,
        )
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
        k_nope, k_rope, v = self._decompress_kv(
            kv,
            infer_state,
            layer_weight,
            False,
            infer_state.total_token_num,
            infer_state.b_seq_len,
            infer_state.max_value_in_b_seq_len,
            infer_state.b1_kv_start_loc,
        )
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
            infer_state.b1_kv_start_loc,
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
        k_nope, k_rope, v = self._decompress_kv(
            kv,
            infer_state,
            layer_weight,
            True,
            infer_state.total_token_num,
            infer_state.b_seq_len,
            infer_state.max_value_in_b_seq_len,
            infer_state.b1_kv_start_loc,
        )
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
            infer_state.b1_kv_start_loc,
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
        return o_tensor

    def _token_gqa_decode_attention_flashattention(
        self, q, infer_state: Deepseek2FlashAttentionStateInfo, layer_weight: Deepseek2TransformerLayerWeight, out=None
    ):
        q_nope, q_rope = q[:, :, : -self.qk_rope_head_dim], q[:, :, -self.qk_rope_head_dim :]
        q_nope = layer_weight.k_b_proj_.bmm(q_nope.transpose(0, 1)).transpose(0, 1)
        kv = infer_state.mem_manager.kv_buffer[self.layer_num_]
        k_rope = kv[:, :, -self.qk_rope_head_dim :].reshape(-1, 1, 1, self.qk_rope_head_dim)
        kv_nope = kv[:, :, : -self.qk_rope_head_dim].reshape(-1, 1, 1, self.kv_lora_rank)
        k_descale, v_descale = None, None
        o_tensor = flash_attn_with_kvcache(
            q=q_rope,
            k_cache=k_rope,
            v_cache=kv_nope,
            qv=q_nope,
            page_table=infer_state.page_table,
            cache_seqlens=infer_state.b_seq_len,
            cu_seqlens_q=infer_state.cu_seqlens_q,
            cu_seqlens_k_new=infer_state.cu_seqlens_k,
            max_seqlen_q=1,
            softmax_scale=self.softmax_scale,
            causal=True,
            window_size=(-1, -1),
            softcap=0.0,
            k_descale=k_descale,
            v_descale=v_descale,
            return_softmax_lse=False,
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
        out = gqa_token_decode_attention_flash_decoding(
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
        return out

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

    def _moe_ffn(
        self, input, infer_state: Deepseek2InferStateInfo, layer_weight: Deepseek2TransformerLayerWeight
    ) -> torch.Tensor:

        hidden_states = input.view(-1, self.embed_dim_)
        num_tokens, hidden_dim = hidden_states.shape

        # if fused_shared_experts is not enabled, compute shared_output
        if self.n_shared_experts is not None and layer_weight.num_fused_shared_experts == 0:
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

        if self.n_shared_experts is not None and layer_weight.num_fused_shared_experts == 0:
            hidden_states.add_(shared_output)

        return hidden_states.view(num_tokens, hidden_dim)

    def _moe_ffn_edp(
        self, input, infer_state: Deepseek2InferStateInfo, layer_weight: Deepseek2TransformerLayerWeight
    ) -> torch.Tensor:

        hidden_states = input
        token_num, hidden_dim = hidden_states.shape
        if self.n_shared_experts is not None:
            shared_output = LlamaTransformerLayerInfer._ffn(self, hidden_states, infer_state, layer_weight)

        router_logits = layer_weight.moe_gate.mm(hidden_states)
        ep_output = layer_weight.experts.experts(
            hidden_states,
            router_logits=router_logits,
            top_k=self.num_experts_per_tok,
            renormalize=self.norm_topk_prob,
            use_grouped_topk=self.n_group,
            topk_group=self.topk_group,
            num_expert_group=self.n_group,
            is_prefill=infer_state.is_prefill,
        )

        if self.n_shared_experts is not None:
            ep_output.add_(shared_output)

        ep_output = ep_output.view(token_num, hidden_dim)
        return ep_output

    def overlap_tpsp_token_forward(
        self,
        input_embdings: torch.Tensor,
        input_embdings1: torch.Tensor,
        infer_state: Deepseek2InferStateInfo,
        infer_state1: Deepseek2InferStateInfo,
        layer_weight: Deepseek2TransformerLayerWeight,
    ):
        if not self.is_moe:
            return super().overlap_tpsp_token_forward(
                input_embdings, input_embdings1, infer_state, infer_state1, layer_weight
            )
        # 0 attention
        _0_input1 = self._att_norm(input_embdings, infer_state, layer_weight)
        _0_cache_kv = self._pre_cache_kv(infer_state, layer_weight)
        _0_q, _0_cache_kv = self._tpsp_get_qkv(_0_input1, _0_cache_kv, infer_state, layer_weight)
        _0_input1 = None
        self._post_cache_kv(_0_cache_kv, infer_state, layer_weight)
        _0_o = self._token_attention_kernel(_0_q, infer_state, layer_weight)
        _0_q = None
        _0_o = self._tpsp_get_o(_0_o, infer_state, layer_weight)
        input_embdings.add_(_0_o.view(-1, self.embed_dim_))
        _0_o = None
        _0_input1 = self._ffn_norm(input_embdings, infer_state, layer_weight)
        _0_router_logits = layer_weight.moe_gate.mm(_0_input1)
        # 1 hook
        if getattr(infer_state1, "hook", None) is not None:
            infer_state1.hook()
            infer_state1.hook = None

        # 0 shared expert
        if self.n_shared_experts is not None:
            _0_shared_output = LlamaTransformerLayerInfer._ffn(self, _0_input1, infer_state, layer_weight)

        # 0 dispatch
        (
            _0_recv_x,
            _0_masked_m,
            _0_topk_idx,
            _0_topk_weight,
            _0_handle,
            _0_hook,
        ) = layer_weight.experts.low_latency_dispatch(_0_input1, _0_router_logits)
        infer_state.hook = _0_hook

        # 1 attention
        _1_input1 = self._att_norm(input_embdings1, infer_state1, layer_weight)
        _1_cache_kv = self._pre_cache_kv(infer_state1, layer_weight)
        _1_q, _1_cache_kv = self._tpsp_get_qkv(_1_input1, _1_cache_kv, infer_state1, layer_weight)
        _1_input1 = None
        self._post_cache_kv(_1_cache_kv, infer_state1, layer_weight)
        _1_o = self._token_attention_kernel(_1_q, infer_state1, layer_weight)
        _1_q = None
        _1_o = self._tpsp_get_o(_1_o, infer_state1, layer_weight)
        input_embdings1.add_(_1_o.view(-1, self.embed_dim_))
        _1_o = None
        _1_input1 = self._ffn_norm(input_embdings1, infer_state1, layer_weight)
        # to do gate and disptatch

        _1_router_logits = layer_weight.moe_gate.mm(_1_input1)
        # 0 hook
        if getattr(infer_state, "hook", None) is not None:
            infer_state.hook()
            infer_state.hook = None

        # 1 shared expert
        if self.n_shared_experts is not None:
            _1_shared_output = LlamaTransformerLayerInfer._ffn(self, _1_input1, infer_state1, layer_weight)

        # 1 dispatch
        (
            _1_recv_x,
            _1_masked_m,
            _1_topk_idx,
            _1_topk_weight,
            _1_handle,
            _1_hook,
        ) = layer_weight.experts.low_latency_dispatch(_1_input1, _1_router_logits)
        infer_state1.hook = _1_hook

        # moe calu
        expected_m = triton.cdiv(
            input_embdings.shape[0] * get_global_world_size() * self.num_experts_per_tok, self.n_routed_experts
        )
        _0_moe_out = layer_weight.experts.masked_group_gemm(_0_recv_x, _0_masked_m, input_embdings.dtype, expected_m)

        # 1 hook
        if getattr(infer_state1, "hook", None) is not None:
            infer_state1.hook()
            infer_state1.hook = None

        # 0 combine
        _0_ffn_out, _0_hook = layer_weight.experts.low_latency_combine(
            _0_moe_out, _0_topk_idx, _0_topk_weight, _0_handle
        )

        infer_state.hook = _0_hook

        # to do moe caclue
        _1_moe_out = layer_weight.experts.masked_group_gemm(_1_recv_x, _1_masked_m, input_embdings1.dtype, expected_m)

        # 0 hook
        if getattr(infer_state, "hook", None) is not None:
            infer_state.hook()
            if self.n_shared_experts is not None:
                _0_ffn_out.add_(_0_shared_output)
            input_embdings.add_(_0_ffn_out.view(-1, self.embed_dim_))
            infer_state.hook = None

        # 1 combine
        _1_ffn_out, _1_hook = layer_weight.experts.low_latency_combine(
            _1_moe_out, _1_topk_idx, _1_topk_weight, _1_handle
        )

        def _1_hook_post():
            _1_hook()
            nonlocal _1_ffn_out
            if self.n_shared_experts is not None:
                _1_ffn_out.add_(_1_shared_output)
            input_embdings1.add_(_1_ffn_out.view(-1, self.embed_dim_))
            return

        infer_state1.hook = _1_hook_post

        return input_embdings, input_embdings1

    def overlap_tpsp_context_forward(
        self,
        input_embdings: torch.Tensor,
        input_embdings1: torch.Tensor,
        infer_state: Deepseek2InferStateInfo,
        infer_state1: Deepseek2InferStateInfo,
        layer_weight: Deepseek2TransformerLayerWeight,
    ):
        if not self.is_moe:
            return super().overlap_tpsp_context_forward(
                input_embdings, input_embdings1, infer_state, infer_state1, layer_weight
            )
        # 0 attention
        _0_input1 = self._att_norm(input_embdings, infer_state, layer_weight)
        _0_cache_kv = self._pre_cache_kv(infer_state, layer_weight)
        _0_q, _0_cache_kv = self._tpsp_get_qkv(_0_input1, _0_cache_kv, infer_state, layer_weight)
        _0_input1 = None
        self._post_cache_kv(_0_cache_kv, infer_state, layer_weight)
        _0_o = self._context_attention_kernel(_0_q, _0_cache_kv, infer_state, layer_weight)
        _0_q = None
        _0_o = self._tpsp_get_o(_0_o, infer_state, layer_weight)
        input_embdings.add_(_0_o.view(-1, self.embed_dim_))
        _0_o = None
        _0_input1 = self._ffn_norm(input_embdings, infer_state, layer_weight)
        _0_router_logits = layer_weight.moe_gate.mm(_0_input1)

        # wait last 1 combine
        if getattr(infer_state1, "hook", None) is not None:
            infer_state1.hook()
            infer_state1.hook = None

        _0_topk_weight, _0_topk_idx, _0_qinput_tensor = layer_weight.experts.select_experts_and_quant_input(
            _0_input1, _0_router_logits
        )
        from deep_ep import Buffer

        _0_overlap_event = Buffer.capture()

        # 1 attention
        _1_input1 = self._att_norm(input_embdings1, infer_state1, layer_weight)
        _1_cache_kv = self._pre_cache_kv(infer_state1, layer_weight)
        _1_q, _1_cache_kv = self._tpsp_get_qkv(_1_input1, _1_cache_kv, infer_state1, layer_weight)
        _1_input1 = None
        self._post_cache_kv(_1_cache_kv, infer_state1, layer_weight)
        _1_o = self._context_attention_kernel(_1_q, _1_cache_kv, infer_state1, layer_weight)
        _1_q = None
        _1_o = self._tpsp_get_o(_1_o, infer_state1, layer_weight)
        input_embdings1.add_(_1_o.view(-1, self.embed_dim_))
        _1_o = None
        _1_input1 = self._ffn_norm(input_embdings1, infer_state1, layer_weight)
        # to do gate and disptatch

        _1_router_logits = layer_weight.moe_gate.mm(_1_input1)

        # 0 dispatch execute
        (
            _0_recv_x,
            _0_recv_topk_idx,
            _0_recv_topk_weight,
            _0_num_recv_tokens_per_expert_list,
            _0_handle,
            _0_hook,
        ) = layer_weight.experts.dispatch(_0_qinput_tensor, _0_topk_idx, _0_topk_weight, overlap_event=_0_overlap_event)
        infer_state.hook = _0_hook

        # wait 0 dispatch
        if getattr(infer_state, "hook", None) is not None:
            infer_state.hook()
            infer_state.hook = None

        _1_topk_weight, _1_topk_idx, _1_qinput_tensor = layer_weight.experts.select_experts_and_quant_input(
            _1_input1, _1_router_logits
        )

        _1_overlap_event = Buffer.capture()

        # 0 shared expert
        if self.n_shared_experts is not None:
            _0_shared_output = LlamaTransformerLayerInfer._ffn(self, _0_input1, infer_state, layer_weight)

        # 1 shared expert
        if self.n_shared_experts is not None:
            _1_shared_output = LlamaTransformerLayerInfer._ffn(self, _1_input1, infer_state1, layer_weight)

        # 0 moe calu
        _0_moe_out = layer_weight.experts.prefilled_group_gemm(
            _0_num_recv_tokens_per_expert_list, _0_recv_x, _0_recv_topk_idx, _0_recv_topk_weight
        )

        # 1 dispatch execute
        (
            _1_recv_x,
            _1_recv_topk_idx,
            _1_recv_topk_weight,
            _1_num_recv_tokens_per_expert_list,
            _1_handle,
            _1_hook,
        ) = layer_weight.experts.dispatch(_1_qinput_tensor, _1_topk_idx, _1_topk_weight, overlap_event=_1_overlap_event)
        infer_state1.hook = _1_hook

        # wait 1 dispatch
        if getattr(infer_state1, "hook", None) is not None:
            infer_state1.hook()
            infer_state1.hook = None

        _0_combine_event = Buffer.capture()
        # 0 combine execute
        _0_ffn_out, _0_hook = layer_weight.experts.combine(_0_moe_out, _0_handle, _0_combine_event)
        infer_state.hook = _0_hook

        # 1 moe calc
        _1_moe_out = layer_weight.experts.prefilled_group_gemm(
            _1_num_recv_tokens_per_expert_list, _1_recv_x, _1_recv_topk_idx, _1_recv_topk_weight
        )

        # wait 0 combine
        if getattr(infer_state, "hook", None) is not None:
            infer_state.hook()
            infer_state.hook = None

        _1_combine_event = Buffer.capture()

        if self.n_shared_experts is not None:
            _0_ffn_out.add_(_0_shared_output)
        input_embdings.add_(_0_ffn_out.view(-1, self.embed_dim_))

        # 1 combine execute
        _1_ffn_out, _1_hook = layer_weight.experts.combine(_1_moe_out, _1_handle, _1_combine_event)

        def _1_hook_post():
            _1_hook()
            nonlocal _1_ffn_out
            if self.n_shared_experts is not None:
                _1_ffn_out.add_(_1_shared_output)
            input_embdings1.add_(_1_ffn_out.view(-1, self.embed_dim_))
            return

        infer_state1.hook = _1_hook_post

        return input_embdings, input_embdings1
