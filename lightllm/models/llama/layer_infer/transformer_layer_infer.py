import torch
import triton
import torch.functional as F
import torch.distributed as dist
import numpy as np
from typing import Tuple
from functools import partial

from lightllm.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight
from lightllm.models.llama.triton_kernel.context_flashattention_nopad import (
    context_attention_fwd,
    context_attention_fwd_ppl_int8kv,
)
from lightllm.models.llama.triton_kernel.token_attention_nopad_att1 import token_att_fwd, token_att_fwd_int8k
from lightllm.models.llama.triton_kernel.token_attention_nopad_softmax import token_softmax_fwd
from lightllm.models.llama.triton_kernel.token_attention_nopad_reduceV import token_att_fwd2, token_att_fwd2_int8v
from lightllm.models.llama.triton_kernel.rmsnorm import rmsnorm_forward
from lightllm.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.common.fused_moe.moe_silu_and_mul import silu_and_mul_fwd

from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.models.llama.flashattention_infer_struct import FlashAttentionStateInfo
from lightllm.models.llama.flashinfer_struct import LlamaFlashInferStateInfo
from lightllm.common.basemodel.triton_kernel.destindex_copy_kv import destindex_copy_kv, destindex_copy_quantize_kv
from lightllm.common.basemodel.triton_kernel.destindex_copy_kv_fp8 import destindex_copy_kv_fp8
from lightllm.common.basemodel import TransformerLayerInferTpl
from lightllm.models.llama.triton_kernel.ppl_quant_copy_kv import destindex_copy_dequantize_kv
from lightllm.distributed.communication_op import all_gather_into_tensor, reduce_scatter_tensor
from lightllm.utils.log_utils import init_logger
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.light_utils import HAS_LIGHTLLM_KERNEL, light_ops
from lightllm.common.basemodel.triton_kernel.q_per_head_fp8_quant import q_per_head_fp8_quant
from lightllm.utils.vllm_utils import HAS_VLLM, vllm_ops

if HAS_VLLM:
    scaled_fp8_quant = vllm_ops.scaled_fp8_quant
else:
    scaled_fp8_quant = None

logger = init_logger(__name__)

from lightllm.utils.sgl_utils import flash_attn_with_kvcache


class LlamaTransformerLayerInfer(TransformerLayerInferTpl):
    """ """

    def __init__(self, layer_num, network_config, mode=[]):
        super().__init__(layer_num, network_config, mode)
        self.eps_ = network_config["rms_norm_eps"]
        self.tp_q_head_num_ = network_config["num_attention_heads"] // self.tp_world_size_
        self.tp_k_head_num_ = max(network_config["num_key_value_heads"] // self.tp_world_size_, 1)
        self.tp_v_head_num_ = max(network_config["num_key_value_heads"] // self.tp_world_size_, 1)
        self.tp_o_head_num_ = self.tp_q_head_num_
        self.head_dim_ = network_config["hidden_size"] // network_config["num_attention_heads"]
        self.embed_dim_ = network_config["hidden_size"]
        self._bind_func()
        return

    def _bind_func(self):
        self._bind_norm()
        self._bind_attention()
        return

    def _bind_norm(self):
        self._att_norm = partial(LlamaTransformerLayerInfer._att_norm, self)
        self._ffn_norm = partial(LlamaTransformerLayerInfer._ffn_norm, self)
        return

    def _bind_attention(self):
        if get_env_start_args().enable_fa3:
            if "offline_calibration_fp8kv" in self.mode:
                self._context_attention_kernel = partial(
                    LlamaTransformerLayerInfer._context_attention_flashattention_fp8, self
                )
                self._token_attention_kernel = partial(
                    LlamaTransformerLayerInfer._token_decode_attention_flashattention_fp8, self
                )
                self._copy_kv_to_mem_cache = partial(LlamaTransformerLayerInfer._copy_kv_to_mem_cache_fp8kv, self)
            elif "export_fp8kv_calibration" in self.mode:
                self._context_attention_kernel = partial(
                    LlamaTransformerLayerInfer._context_attention_flashattention, self
                )
                self._token_attention_kernel = partial(
                    LlamaTransformerLayerInfer._token_decode_attention_flashattention, self
                )
                self._copy_kv_to_mem_cache = partial(
                    LlamaTransformerLayerInfer._copy_kv_to_mem_cache_with_calibration, self
                )
            elif not self.mode:
                self._context_attention_kernel = partial(
                    LlamaTransformerLayerInfer._context_attention_flashattention, self
                )
                self._token_attention_kernel = partial(
                    LlamaTransformerLayerInfer._token_decode_attention_flashattention, self
                )
                self._copy_kv_to_mem_cache = partial(LlamaTransformerLayerInfer._copy_kv_to_mem_cache_normal, self)
            else:
                raise Exception(f"Unsupported mode for fa3 backend: {self.mode}")
            return
        elif get_env_start_args().enable_flashinfer_prefill:
            self._context_attention_kernel = partial(
                LlamaTransformerLayerInfer._context_attention_flashinfer_kernel, self
            )
        else:
            self._context_attention_kernel = partial(LlamaTransformerLayerInfer._context_attention_kernel, self)
        if "ppl_int8kv" in self.mode:
            self._token_attention_kernel = partial(LlamaTransformerLayerInfer._token_decode_attention_ppl_int8kv, self)
            self._copy_kv_to_mem_cache = partial(LlamaTransformerLayerInfer._copy_kv_to_mem_cache_ppl_int8kv, self)
            self._context_attention_kernel = partial(
                LlamaTransformerLayerInfer._context_attention_kernel_ppl_int8kv, self
            )
        elif "ppl_int8kv_flashdecoding" in self.mode:
            self._token_attention_kernel = partial(
                LlamaTransformerLayerInfer._token_decode_attention_ppl_int8kv_flashdecoding, self
            )
            self._copy_kv_to_mem_cache = partial(LlamaTransformerLayerInfer._copy_kv_to_mem_cache_ppl_int8kv, self)
            self._context_attention_kernel = partial(
                LlamaTransformerLayerInfer._context_attention_kernel_ppl_int8kv, self
            )
        elif "ppl_int4kv_flashdecoding" in self.mode:
            self._token_attention_kernel = partial(
                LlamaTransformerLayerInfer._token_decode_attention_ppl_int4kv_flashdecoding, self
            )
            self._copy_kv_to_mem_cache = partial(LlamaTransformerLayerInfer._copy_kv_to_mem_cache_ppl_int4kv, self)
        elif "ppl_fp16" in self.mode:
            self._token_attention_kernel = partial(LlamaTransformerLayerInfer._token_decode_attention_ppl_fp16, self)
            self._copy_kv_to_mem_cache = partial(LlamaTransformerLayerInfer._copy_kv_to_mem_cache_normal, self)
        elif "ppl_fp16_flashdecoding" in self.mode:
            self._token_attention_kernel = partial(
                LlamaTransformerLayerInfer._token_decode_attention_ppl_fp16_flashdecoding, self
            )
            self._copy_kv_to_mem_cache = partial(LlamaTransformerLayerInfer._copy_kv_to_mem_cache_normal, self)
        elif "triton_int8kv" in self.mode:
            self._token_attention_kernel = partial(LlamaTransformerLayerInfer._token_decode_attention_int8kv, self)
            self._copy_kv_to_mem_cache = partial(LlamaTransformerLayerInfer._copy_kv_to_mem_cache_int8kv, self)
        elif "offline_calibration_fp8kv" in self.mode:
            assert get_env_start_args().enable_flashinfer_prefill and get_env_start_args().enable_flashinfer_decode
            self._copy_kv_to_mem_cache = partial(LlamaTransformerLayerInfer._copy_kv_to_mem_cache_fp8kv, self)
            self._context_attention_kernel = partial(
                LlamaTransformerLayerInfer._context_attention_flashinfer_kernel_fp8, self
            )
            self._token_attention_kernel = partial(
                LlamaTransformerLayerInfer._token_decode_attention_flashinfer_fp8, self
            )
        elif "triton_flashdecoding" in self.mode:
            self._token_attention_kernel = partial(
                LlamaTransformerLayerInfer._token_decode_attention_flashdecoding, self
            )
            self._copy_kv_to_mem_cache = partial(LlamaTransformerLayerInfer._copy_kv_to_mem_cache_normal, self)
        elif "triton_gqa_attention" in self.mode:
            self._token_attention_kernel = partial(LlamaTransformerLayerInfer._token_decode_gqa_attention_normal, self)
            self._copy_kv_to_mem_cache = partial(LlamaTransformerLayerInfer._copy_kv_to_mem_cache_normal, self)
        elif "triton_gqa_flashdecoding" in self.mode:
            self._token_attention_kernel = partial(
                LlamaTransformerLayerInfer._token_decode_attention_gqa_flashdecoding, self
            )
            self._copy_kv_to_mem_cache = partial(LlamaTransformerLayerInfer._copy_kv_to_mem_cache_normal, self)
        elif "triton_gqa_flashdecoding_vsm" in self.mode:
            self._token_attention_kernel = partial(
                LlamaTransformerLayerInfer._token_decode_attention_gqa_flashdecoding_vsm, self
            )
            self._copy_kv_to_mem_cache = partial(LlamaTransformerLayerInfer._copy_kv_to_mem_cache_normal, self)
        elif "export_fp8kv_calibration" in self.mode:
            self._token_attention_kernel = partial(LlamaTransformerLayerInfer._token_decode_attention_flashinfer, self)
            self._copy_kv_to_mem_cache = partial(
                LlamaTransformerLayerInfer._copy_kv_to_mem_cache_with_calibration, self
            )
        elif not self.mode:
            if get_env_start_args().enable_flashinfer_decode:
                self._token_attention_kernel = partial(
                    LlamaTransformerLayerInfer._token_decode_attention_flashinfer, self
                )
            else:
                self._token_attention_kernel = partial(LlamaTransformerLayerInfer._token_decode_attention_normal, self)
            self._copy_kv_to_mem_cache = partial(LlamaTransformerLayerInfer._copy_kv_to_mem_cache_normal, self)
        else:
            raise Exception(f"Unsupported mode: {self.mode}")

        return

    def _att_norm(
        self, input, infer_state: LlamaInferStateInfo, layer_weight: LlamaTransformerLayerWeight
    ) -> torch.Tensor:
        out = self.alloc_tensor(input.shape, input.dtype)
        rmsnorm_forward(input, weight=layer_weight.att_norm_weight_.weight, eps=self.eps_, out=out)
        return out

    def _ffn_norm(
        self, input, infer_state: LlamaInferStateInfo, layer_weight: LlamaTransformerLayerWeight
    ) -> torch.Tensor:
        out = self.alloc_tensor(input.shape, input.dtype)
        rmsnorm_forward(input, weight=layer_weight.ffn_norm_weight_.weight, eps=self.eps_, out=out)
        return out

    def _get_qkv(
        self, input, infer_state: LlamaInferStateInfo, layer_weight: LlamaTransformerLayerWeight
    ) -> torch.Tensor:
        q = layer_weight.q_proj.mm(input)
        cache_kv = self._pre_cache_kv(infer_state=infer_state, layer_weight=layer_weight)
        cache_kv = layer_weight.kv_proj.mm(
            input, out=cache_kv.view(-1, (self.tp_k_head_num_ + self.tp_v_head_num_) * self.head_dim_)
        ).view(-1, (self.tp_k_head_num_ + self.tp_v_head_num_), self.head_dim_)

        rotary_emb_fwd(
            q.view(-1, self.tp_q_head_num_, self.head_dim_),
            cache_kv[:, 0 : self.tp_k_head_num_, :],
            infer_state.position_cos,
            infer_state.position_sin,
        )
        return q, cache_kv

    def _tpsp_get_qkv(
        self, input, infer_state: LlamaInferStateInfo, layer_weight: LlamaTransformerLayerWeight
    ) -> torch.Tensor:
        if self.tp_world_size_ > 1:
            sp_token_num, hidden_dim = input.shape
            gather_input = self.alloc_tensor(
                (sp_token_num * self.tp_world_size_, hidden_dim), dtype=input.dtype, device=input.device
            )
            all_gather_into_tensor(gather_input, input, group=infer_state.dist_group, async_op=False)
            input = gather_input[0 : len(infer_state.position_cos), :]

        q = layer_weight.q_proj.mm(input)
        cache_kv = self._pre_cache_kv(infer_state=infer_state, layer_weight=layer_weight)
        cache_kv = layer_weight.kv_proj.mm(
            input, out=cache_kv.view(-1, (self.tp_k_head_num_ + self.tp_v_head_num_) * self.head_dim_)
        ).view(-1, (self.tp_k_head_num_ + self.tp_v_head_num_), self.head_dim_)

        rotary_emb_fwd(
            q.view(-1, self.tp_q_head_num_, self.head_dim_),
            cache_kv[:, 0 : self.tp_k_head_num_, :],
            infer_state.position_cos,
            infer_state.position_sin,
        )
        return q, cache_kv

    def _context_attention_flashinfer_kernel_fp8(
        self, q, kv, infer_state: LlamaFlashInferStateInfo, layer_weight, out=None
    ) -> torch.Tensor:
        o_tensor = self.alloc_tensor(q.shape, q.dtype) if out is None else out
        kv = infer_state.mem_manager.kv_buffer[self.layer_num_]
        kv = kv.unsqueeze(1)
        k = kv[:, :, : self.tp_k_head_num_, :].view(torch.float8_e4m3fn)
        v = kv[:, :, self.tp_k_head_num_ :, :].view(torch.float8_e4m3fn)
        offline_scales = infer_state.mem_manager.scales_list
        k_descale = offline_scales[self.layer_num_][0] if offline_scales is not None else None
        v_descale = offline_scales[self.layer_num_][1] if offline_scales is not None else None
        infer_state.prefill_wrapper.run(
            q.view(q.shape[0], -1, self.head_dim_),
            (k, v),
            k_scale=k_descale,
            v_scale=v_descale,
            out=o_tensor.view(q.shape[0], -1, self.head_dim_),
        )
        return o_tensor

    def _context_attention_flashinfer_kernel(
        self, q, kv, infer_state: LlamaFlashInferStateInfo, layer_weight, out=None
    ) -> torch.Tensor:
        o_tensor = self.alloc_tensor(q.shape, q.dtype) if out is None else out
        kv = infer_state.mem_manager.kv_buffer[self.layer_num_]
        kv = kv.unsqueeze(1)
        infer_state.prefill_wrapper.run(
            q.view(q.shape[0], -1, self.head_dim_),
            (kv[:, :, : self.tp_k_head_num_, :], kv[:, :, self.tp_k_head_num_ :, :]),
            out=o_tensor.view(q.shape[0], -1, self.head_dim_),
        )
        return o_tensor

    def _context_attention_kernel(
        self, q, kv, infer_state: LlamaInferStateInfo, layer_weight, out=None
    ) -> torch.Tensor:
        o_tensor = self.alloc_tensor(q.shape, q.dtype) if out is None else out
        kv = infer_state.mem_manager.kv_buffer[self.layer_num_]
        context_attention_fwd(
            q.view(-1, self.tp_q_head_num_, self.head_dim_),
            kv[:, 0 : self.tp_k_head_num_, :],
            kv[:, self.tp_k_head_num_ : self.tp_k_head_num_ + self.tp_v_head_num_, :],
            o_tensor.view(-1, self.tp_q_head_num_, self.head_dim_),
            infer_state.b_req_idx,
            infer_state.b_start_loc,
            infer_state.b_seq_len,
            infer_state.b_ready_cache_len,
            infer_state.max_len_in_batch,
            infer_state.req_manager.req_to_token_indexs,
        )
        return o_tensor

    def _context_attention_kernel_ppl_int8kv(
        self, q, kv, infer_state: LlamaInferStateInfo, layer_weight, out=None
    ) -> torch.Tensor:
        o_tensor = self.alloc_tensor(q.shape, q.dtype) if out is None else out
        batch_size = infer_state.b_seq_len.shape[0]
        kv = infer_state.mem_manager.kv_buffer[self.layer_num_]
        kv_scale = infer_state.mem_manager.scale_buffer[self.layer_num_]
        max_seq_len = infer_state.max_seq_len
        kv_dequant = self.alloc_tensor(
            (batch_size, kv.shape[1], max_seq_len, kv.shape[2]), device=q.device, dtype=q.dtype
        )
        destindex_copy_dequantize_kv(
            kv,
            kv_scale,
            infer_state.req_manager.req_to_token_indexs,
            infer_state.b_seq_len,
            infer_state.b_req_idx,
            max_seq_len,
            kv_dequant,
        )
        context_attention_fwd_ppl_int8kv(
            q.view(-1, self.tp_q_head_num_, self.head_dim_),
            kv_dequant[:, 0 : self.tp_k_head_num_, :, :],
            kv_dequant[:, self.tp_k_head_num_ : self.tp_k_head_num_ + self.tp_v_head_num_, :, :],
            o_tensor.view(-1, self.tp_q_head_num_, self.head_dim_),
            infer_state.b_start_loc,
            infer_state.b_seq_len,
            infer_state.max_len_in_batch,
            infer_state.b_ready_cache_len,
        )
        return o_tensor

    def _context_attention_flashattention(self, q, kv, infer_state: FlashAttentionStateInfo, layer_weight, out=None):
        cache_k = infer_state.mem_manager.kv_buffer[self.layer_num_][:, 0 : self.tp_k_head_num_, :].reshape(
            -1, 1, self.tp_k_head_num_, self.head_dim_
        )
        cache_v = infer_state.mem_manager.kv_buffer[self.layer_num_][
            :, self.tp_k_head_num_ : self.tp_k_head_num_ + self.tp_v_head_num_, :
        ].reshape(-1, 1, self.tp_v_head_num_, self.head_dim_)
        q = q.reshape(-1, self.tp_q_head_num_, self.head_dim_)
        k_descale, v_descale = None, None  # disable quantization
        Lq = q.shape[-1]
        sm_scale = 1.0 / (Lq ** 0.5)
        o = flash_attn_with_kvcache(
            q=q,
            k_cache=cache_k,
            v_cache=cache_v,
            page_table=infer_state.page_table,
            cache_seqlens=infer_state.b_seq_len,
            cu_seqlens_q=infer_state.cu_seqlens_q,
            cu_seqlens_k_new=infer_state.cu_seqlens_k,
            max_seqlen_q=infer_state.q_max_seq_len,
            softmax_scale=sm_scale,
            causal=True,
            window_size=(-1, -1),
            softcap=0.0,
            k_descale=k_descale,
            v_descale=v_descale,
            return_softmax_lse=False,
        )
        return o

    def _context_attention_flashattention_fp8(
        self, q, kv, infer_state: FlashAttentionStateInfo, layer_weight, out=None
    ):
        q, q_scale = q_per_head_fp8_quant(
            q.view(q.shape[0], self.tp_k_head_num_, -1),
            infer_state.b_seq_len,
            infer_state.cu_seqlens_q,
            infer_state.q_scale,
            infer_state.token_batch_ids,
        )
        cache_k = (
            (infer_state.mem_manager.kv_buffer[self.layer_num_][:, : self.tp_k_head_num_, :])
            .reshape(-1, 1, self.tp_k_head_num_, self.head_dim_)
            .view(torch.float8_e4m3fn)
        )
        cache_v = (
            (
                infer_state.mem_manager.kv_buffer[self.layer_num_][
                    :, self.tp_k_head_num_ : self.tp_k_head_num_ + self.tp_v_head_num_, :
                ]
            )
            .reshape(-1, 1, self.tp_v_head_num_, self.head_dim_)
            .view(torch.float8_e4m3fn)
        )
        o = flash_attn_with_kvcache(
            q=q.view(-1, self.tp_q_head_num_, self.head_dim_),
            k_cache=cache_k,
            v_cache=cache_v,
            page_table=infer_state.page_table,
            cache_seqlens=infer_state.b_seq_len,
            cu_seqlens_q=infer_state.cu_seqlens_q,
            cu_seqlens_k_new=infer_state.cu_seqlens_k,
            max_seqlen_q=infer_state.q_max_seq_len,
            causal=True,
            window_size=(-1, -1),
            softcap=0.0,
            q_descale=q_scale,
            k_descale=infer_state.k_descale[self.layer_num_],
            v_descale=infer_state.v_descale[self.layer_num_],
            return_softmax_lse=False,
        )
        return o

    def _get_o(
        self, input, infer_state: LlamaInferStateInfo, layer_weight: LlamaTransformerLayerWeight
    ) -> torch.Tensor:
        input = input.view(-1, self.tp_o_head_num_ * self.head_dim_)
        o_tensor = layer_weight.o_proj.mm(input)
        return o_tensor

    def _tpsp_get_o(
        self, input, infer_state: LlamaInferStateInfo, layer_weight: LlamaTransformerLayerWeight
    ) -> torch.Tensor:
        input = input.view(-1, self.tp_o_head_num_ * self.head_dim_)
        dest_size = triton.cdiv(input.shape[0], self.tp_world_size_) * self.tp_world_size_
        o_tensor = self.alloc_tensor((dest_size, self.embed_dim_), dtype=input.dtype, device=input.device)
        layer_weight.o_proj.mm(input, out=o_tensor[0 : len(infer_state.position_cos), :])

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

    def _ffn(self, input, infer_state: LlamaInferStateInfo, layer_weight: LlamaTransformerLayerWeight) -> torch.Tensor:
        input = input.view(-1, self.embed_dim_)
        up_gate_out = layer_weight.gate_up_proj.mm(input)
        ffn1_out = self.alloc_tensor((input.size(0), up_gate_out.size(1) // 2), input.dtype)
        silu_and_mul_fwd(up_gate_out, ffn1_out)
        input = None
        up_gate_out = None
        ffn2_out = layer_weight.down_proj.mm(ffn1_out)
        ffn1_out = None
        return ffn2_out

    def _tpsp_ffn(
        self, input, infer_state: LlamaInferStateInfo, layer_weight: LlamaTransformerLayerWeight
    ) -> torch.Tensor:
        input = input.view(-1, self.embed_dim_)
        if self.tp_world_size_ > 1:
            sp_token_num, hidden_dim = input.shape
            gather_input = self.alloc_tensor(
                (sp_token_num * self.tp_world_size_, hidden_dim), dtype=input.dtype, device=input.device
            )
            all_gather_into_tensor(gather_input, input, group=infer_state.dist_group, async_op=False)
            input = gather_input

        up_gate_out = layer_weight.gate_up_proj.mm(input)
        ffn1_out = self.alloc_tensor((input.size(0), up_gate_out.size(1) // 2), input.dtype)
        silu_and_mul_fwd(up_gate_out, ffn1_out)
        input = None
        up_gate_out = None
        ffn2_out = layer_weight.down_proj.mm(ffn1_out)
        ffn1_out = None
        if self.tp_world_size_ > 1:
            sp_token_num = ffn2_out.shape[0] // self.tp_world_size_
            reduce_o_tensor = self.alloc_tensor(
                (sp_token_num, self.embed_dim_), dtype=ffn2_out.dtype, device=ffn2_out.device
            )
            reduce_scatter_tensor(
                reduce_o_tensor, ffn2_out, op=dist.ReduceOp.SUM, group=infer_state.dist_group, async_op=False
            )
            ffn2_out = reduce_o_tensor
        return ffn2_out

    # # keep code
    # def _ffn(self, input, infer_state: LlamaInferStateInfo, layer_weight: LlamaTransformerLayerWeight)->torch.Tensor:
    #     gate_up_out = torch.mm(input.view(-1, self.embed_dim_), layer_weight.gate_up_proj)
    #     size = gate_up_out.shape[1]
    #     gate_out, up_out = gate_up_out[:, 0: size // 2], gate_up_out[:, size // 2:]
    #     torch.nn.functional.silu(gate_out, inplace=True)
    #     gate_out.mul_(up_out)
    #     input = None
    #     ffn2_out = torch.mm(gate_out, layer_weight.down_proj)
    #     gate_out, up_out = None, None
    #     return ffn2_out

    def _copy_kv_to_mem_cache_normal(self, buffer, mem_index, mem_manager):
        destindex_copy_kv(buffer, mem_index, mem_manager.kv_buffer[self.layer_num_])
        return

    def _copy_kv_to_mem_cache_with_calibration(self, buffer, mem_index, mem_manager):
        destindex_copy_kv(buffer, mem_index, mem_manager.kv_buffer[self.layer_num_])
        mem_manager.update_calibration_data(buffer, self.layer_num_)
        return

    def _copy_kv_to_mem_cache_int8kv(self, buffer, mem_index, mem_manager):
        destindex_copy_quantize_kv(
            buffer, mem_index, mem_manager.kv_buffer[self.layer_num_], mem_manager.scale_buffer[self.layer_num_]
        )
        return

    def _copy_kv_to_mem_cache_fp8kv(self, buffer, mem_index, mem_manager):
        scales = mem_manager.scales
        destindex_copy_kv_fp8(
            buffer,
            mem_index,
            scales[self.layer_num_] if scales is not None else None,
            mem_manager.kv_buffer[self.layer_num_].view(torch.float8_e4m3fn),
        )
        return

    def _copy_kv_to_mem_cache_ppl_int8kv(self, buffer, mem_index, mem_manager):
        from lightllm.models.llama.triton_kernel.ppl_quant_copy_kv import destindex_copy_quantize_kv

        destindex_copy_quantize_kv(
            buffer, mem_index, mem_manager.kv_buffer[self.layer_num_], mem_manager.scale_buffer[self.layer_num_]
        )
        return

    def _copy_kv_to_mem_cache_ppl_int4kv(self, buffer, mem_index, mem_manager):
        from lightllm.models.llama.triton_kernel.ppl_int4kv_copy_kv import destindex_copy_int4kv

        destindex_copy_int4kv(
            buffer, mem_index, mem_manager.kv_buffer[self.layer_num_], mem_manager.scale_buffer[self.layer_num_]
        )
        return

    def _token_decode_attention_flashinfer_fp8(self, q, infer_state: LlamaFlashInferStateInfo, layer_weight, out=None):
        batch_size = infer_state.batch_size
        calcu_shape1 = (batch_size, self.tp_q_head_num_, self.head_dim_)

        o_tensor = self.alloc_tensor(q.shape, q.dtype) if out is None else out
        kv = infer_state.mem_manager.kv_buffer[self.layer_num_].unsqueeze(1)
        k = kv[:, :, : self.tp_k_head_num_, :].view(torch.float8_e4m3fn)
        v = kv[:, :, self.tp_k_head_num_ :, :].view(torch.float8_e4m3fn)
        offline_scales = infer_state.mem_manager.scales_list
        k_descale = offline_scales[self.layer_num_][0] if offline_scales is not None else None
        v_descale = offline_scales[self.layer_num_][1] if offline_scales is not None else None
        infer_state.decode_wrapper.run(
            q.view(calcu_shape1),
            (k, v),
            k_scale=k_descale,
            v_scale=v_descale,
            out=o_tensor.view(calcu_shape1),
        )
        return o_tensor

    def _token_decode_attention_flashinfer(self, q, infer_state: LlamaFlashInferStateInfo, layer_weight, out=None):
        batch_size = infer_state.batch_size
        calcu_shape1 = (batch_size, self.tp_q_head_num_, self.head_dim_)

        o_tensor = self.alloc_tensor(q.shape, q.dtype) if out is None else out
        kv = infer_state.mem_manager.kv_buffer[self.layer_num_].unsqueeze(1)
        infer_state.decode_wrapper.run(
            q.view(calcu_shape1),
            (kv[:, :, : self.tp_k_head_num_, :], kv[:, :, self.tp_k_head_num_ :, :]),
            out=o_tensor.view(calcu_shape1),
        )
        return o_tensor

    def _token_decode_attention_normal(self, q, infer_state: LlamaInferStateInfo, layer_weight, out=None):
        total_token_num = infer_state.total_token_num
        batch_size = infer_state.batch_size
        calcu_shape1 = (batch_size, self.tp_q_head_num_, self.head_dim_)

        att_m_tensor = self.alloc_tensor((self.tp_q_head_num_, total_token_num), torch.float32)

        token_att_fwd(
            q.view(calcu_shape1),
            infer_state.mem_manager.kv_buffer[self.layer_num_][:, 0 : self.tp_k_head_num_, :],
            att_m_tensor,
            infer_state.req_manager.req_to_token_indexs,
            infer_state.b_req_idx,
            infer_state.b_start_loc,
            infer_state.b_seq_len,
            infer_state.max_len_in_batch,
        )

        o_tensor = self.alloc_tensor(q.shape, q.dtype) if out is None else out
        from lightllm.models.llama.triton_kernel.token_attention_softmax_and_reducev import (
            token_softmax_reducev_fwd,
        )

        token_softmax_reducev_fwd(
            att_m_tensor,
            infer_state.mem_manager.kv_buffer[self.layer_num_][
                :, self.tp_k_head_num_ : self.tp_k_head_num_ + self.tp_v_head_num_, :
            ],
            o_tensor.view(calcu_shape1),
            infer_state.req_manager.req_to_token_indexs,
            infer_state.b_req_idx,
            infer_state.b_start_loc,
            infer_state.b_seq_len,
        )
        return o_tensor

    def _token_decode_gqa_attention_normal(self, q, infer_state: LlamaInferStateInfo, layer_weight, out=None):
        batch_size = infer_state.batch_size
        calcu_shape1 = (batch_size, self.tp_q_head_num_, self.head_dim_)
        # 对 gqa模型进行推理优化的代码
        from ..triton_kernel.gqa_decode_flashattention_nopad import gqa_decode_attention_fwd

        o_tensor = self.alloc_tensor(q.shape, q.dtype) if out is None else out
        gqa_decode_attention_fwd(
            q.view(calcu_shape1),
            infer_state.mem_manager.kv_buffer[self.layer_num_][:, 0 : self.tp_k_head_num_, :],
            infer_state.mem_manager.kv_buffer[self.layer_num_][
                :, self.tp_k_head_num_ : self.tp_k_head_num_ + self.tp_v_head_num_, :
            ],
            o_tensor.view(calcu_shape1),
            infer_state.req_manager.req_to_token_indexs,
            infer_state.b_req_idx,
            infer_state.b_seq_len,
        )
        return o_tensor

    def _token_decode_attention_int8kv(self, q, infer_state: LlamaInferStateInfo, layer_weight, out=None):
        total_token_num = infer_state.total_token_num
        batch_size = infer_state.batch_size
        calcu_shape1 = (batch_size, self.tp_q_head_num_, self.head_dim_)
        att_m_tensor = self.alloc_tensor((self.tp_q_head_num_, total_token_num), q.dtype)
        token_att_fwd_int8k(
            q.view(calcu_shape1),
            infer_state.mem_manager.kv_buffer[self.layer_num_][:, 0 : self.tp_k_head_num_, :],
            infer_state.mem_manager.scale_buffer[self.layer_num_][:, 0 : self.tp_k_head_num_, :],
            att_m_tensor,
            infer_state.req_manager.req_to_token_indexs,
            infer_state.b_req_idx,
            infer_state.b_start_loc,
            infer_state.b_seq_len,
            infer_state.max_len_in_batch,
        )

        prob = self.alloc_tensor(att_m_tensor.shape, att_m_tensor.dtype)
        token_softmax_fwd(
            att_m_tensor, infer_state.b_start_loc, infer_state.b_seq_len, prob, infer_state.max_len_in_batch
        )
        att_m_tensor = None

        o_tensor = self.alloc_tensor(q.shape, q.dtype) if out is None else out
        token_att_fwd2_int8v(
            prob,
            infer_state.mem_manager.kv_buffer[self.layer_num_][
                :, self.tp_k_head_num_ : self.tp_k_head_num_ + self.tp_v_head_num_, :
            ],
            infer_state.mem_manager.scale_buffer[self.layer_num_][
                :, self.tp_k_head_num_ : self.tp_k_head_num_ + self.tp_v_head_num_, :
            ],
            o_tensor.view(calcu_shape1),
            infer_state.req_manager.req_to_token_indexs,
            infer_state.b_req_idx,
            infer_state.b_start_loc,
            infer_state.b_seq_len,
            infer_state.max_len_in_batch,
        )
        prob = None
        return o_tensor

    def _token_decode_attention_flashdecoding(self, q, infer_state: LlamaInferStateInfo, layer_weight, out=None):
        from lightllm.models.llama.triton_kernel.flash_decoding import token_decode_attention_flash_decoding

        cache_k = infer_state.mem_manager.kv_buffer[self.layer_num_][:, 0 : self.tp_k_head_num_, :]
        cache_v = infer_state.mem_manager.kv_buffer[self.layer_num_][
            :, self.tp_k_head_num_ : self.tp_k_head_num_ + self.tp_v_head_num_, :
        ]
        return token_decode_attention_flash_decoding(
            q,
            infer_state,
            self.tp_q_head_num_,
            self.head_dim_,
            cache_k,
            cache_v,
            out=out,
            alloc_tensor_func=self.alloc_tensor,
        )

    def _token_decode_attention_gqa_flashdecoding(self, q, infer_state: LlamaInferStateInfo, layer_weight, out=None):
        # 对 gqa 模型进行推理优化的代码
        from ..triton_kernel.gqa_flash_decoding import gqa_token_decode_attention_flash_decoding

        cache_k = infer_state.mem_manager.kv_buffer[self.layer_num_][:, 0 : self.tp_k_head_num_, :]
        cache_v = infer_state.mem_manager.kv_buffer[self.layer_num_][
            :, self.tp_k_head_num_ : self.tp_k_head_num_ + self.tp_v_head_num_, :
        ]
        return gqa_token_decode_attention_flash_decoding(
            q,
            infer_state,
            self.tp_q_head_num_,
            self.head_dim_,
            cache_k,
            cache_v,
            out=out,
            alloc_tensor_func=self.alloc_tensor,
        )

    def _token_decode_attention_ppl_int8kv(self, q, infer_state: LlamaInferStateInfo, layer_weight, out=None):
        batch_size = infer_state.batch_size
        calcu_shape1 = (batch_size, self.tp_q_head_num_, self.head_dim_)
        o_tensor = self.alloc_tensor(q.shape, q.dtype) if out is None else out

        # group_int8kv_decode_attention(at::Tensor o, at::Tensor q, at::Tensor k, at::Tensor k_s,  at::Tensor v,
        # at::Tensor v_s, at::Tensor b_loc, at::Tensor b_seq_len, int max_len_in_batch)
        light_ops.group8_int8kv_decode_attention(
            o_tensor.view(calcu_shape1),
            q.view(calcu_shape1),
            infer_state.mem_manager.kv_buffer[self.layer_num_][:, 0 : self.tp_k_head_num_, :],
            infer_state.mem_manager.scale_buffer[self.layer_num_][:, 0 : self.tp_k_head_num_, :],
            infer_state.mem_manager.kv_buffer[self.layer_num_][
                :, self.tp_k_head_num_ : self.tp_k_head_num_ + self.tp_v_head_num_, :
            ],
            infer_state.mem_manager.scale_buffer[self.layer_num_][
                :, self.tp_k_head_num_ : self.tp_k_head_num_ + self.tp_v_head_num_, :
            ],
            infer_state.req_manager.req_to_token_indexs,
            infer_state.b_req_idx,
            infer_state.b_seq_len,
            infer_state.max_len_in_batch,
        )

        return o_tensor

    def _token_decode_attention_ppl_fp16(self, q, infer_state: LlamaInferStateInfo, layer_weight, out=None):
        batch_size = infer_state.batch_size
        calcu_shape1 = (batch_size, self.tp_q_head_num_, self.head_dim_)
        o_tensor = self.alloc_tensor(q.shape, q.dtype) if out is None else out
        from lightllm_ppl_fp16_kernel import fp16_decode_attention

        # group_int8kv_decode_attention(at::Tensor o, at::Tensor q, at::Tensor k, at::Tensor k_s,
        # at::Tensor v,  at::Tensor v_s, at::Tensor b_loc, at::Tensor b_seq_len, int max_len_in_batch)
        fp16_decode_attention(
            o_tensor.view(calcu_shape1),
            1.0 / (self.head_dim_ ** 0.5),
            q.view(calcu_shape1),
            infer_state.mem_manager.kv_buffer[self.layer_num_][:, 0 : self.tp_k_head_num_, :],
            infer_state.mem_manager.kv_buffer[self.layer_num_][
                :, self.tp_k_head_num_ : self.tp_k_head_num_ + self.tp_v_head_num_, :
            ],
            infer_state.req_manager.req_to_token_indexs,
            infer_state.b_req_idx,
            infer_state.b_seq_len,
            infer_state.max_len_in_batch,
        )

        return o_tensor

    def _token_decode_attention_ppl_fp16_flashdecoding(
        self, q, infer_state: LlamaInferStateInfo, layer_weight, out=None
    ):
        from lightllm.models.llama.triton_kernel.ppl_fp16_flash_decoding import token_decode_attention_flash_decoding

        cache_k = infer_state.mem_manager.kv_buffer[self.layer_num_][:, 0 : self.tp_k_head_num_, :]
        cache_v = infer_state.mem_manager.kv_buffer[self.layer_num_][
            :, self.tp_k_head_num_ : self.tp_k_head_num_ + self.tp_v_head_num_, :
        ]
        return token_decode_attention_flash_decoding(
            q,
            infer_state,
            self.tp_q_head_num_,
            self.head_dim_,
            cache_k,
            cache_v,
            out=out,
            alloc_tensor_func=self.alloc_tensor,
        )

    def _token_decode_attention_ppl_int8kv_flashdecoding(
        self, q, infer_state: LlamaInferStateInfo, layer_weight, out=None
    ):
        from lightllm.models.llama.triton_kernel.ppl_int8kv_flash_decoding import token_decode_attention_flash_decoding

        cache_k = infer_state.mem_manager.kv_buffer[self.layer_num_][:, 0 : self.tp_k_head_num_, :]
        cache_k_scale = infer_state.mem_manager.scale_buffer[self.layer_num_][:, 0 : self.tp_k_head_num_, :]
        cache_v = infer_state.mem_manager.kv_buffer[self.layer_num_][
            :, self.tp_k_head_num_ : self.tp_k_head_num_ + self.tp_v_head_num_, :
        ]
        cache_v_scale = infer_state.mem_manager.scale_buffer[self.layer_num_][
            :, self.tp_k_head_num_ : self.tp_k_head_num_ + self.tp_v_head_num_, :
        ]
        return token_decode_attention_flash_decoding(
            q,
            infer_state,
            self.tp_q_head_num_,
            self.head_dim_,
            cache_k,
            cache_k_scale,
            cache_v,
            cache_v_scale,
            out=out,
            alloc_tensor_func=self.alloc_tensor,
        )

    def _token_decode_attention_ppl_int4kv_flashdecoding(
        self, q, infer_state: LlamaInferStateInfo, layer_weight, out=None
    ):
        from lightllm.models.llama.triton_kernel.ppl_int4kv_flash_decoding import token_decode_attention_flash_decoding

        cache_k = infer_state.mem_manager.kv_buffer[self.layer_num_][:, 0 : self.tp_k_head_num_, :]
        cache_k_scale = infer_state.mem_manager.scale_buffer[self.layer_num_][:, 0 : self.tp_k_head_num_, :]
        cache_v = infer_state.mem_manager.kv_buffer[self.layer_num_][
            :, self.tp_k_head_num_ : self.tp_k_head_num_ + self.tp_v_head_num_, :
        ]
        cache_v_scale = infer_state.mem_manager.scale_buffer[self.layer_num_][
            :, self.tp_k_head_num_ : self.tp_k_head_num_ + self.tp_v_head_num_, :
        ]
        return token_decode_attention_flash_decoding(
            q,
            infer_state,
            self.tp_q_head_num_,
            self.head_dim_,
            cache_k,
            cache_k_scale,
            cache_v,
            cache_v_scale,
            out=out,
            alloc_tensor_func=self.alloc_tensor,
        )

    def _token_decode_attention_gqa_flashdecoding_vsm(
        self, q, infer_state: LlamaInferStateInfo, layer_weight, out=None
    ):
        from lightllm.models.llama.triton_kernel.gqa_flash_decoding_vsm import (
            gqa_token_decode_attention_flash_decoding_vsm,
        )

        cache_k = infer_state.mem_manager.kv_buffer[self.layer_num_][:, 0 : self.tp_k_head_num_, :]
        cache_v = infer_state.mem_manager.kv_buffer[self.layer_num_][
            :, self.tp_k_head_num_ : self.tp_k_head_num_ + self.tp_v_head_num_, :
        ]
        q_shape = (infer_state.batch_size, self.tp_q_head_num_, self.head_dim_)
        return gqa_token_decode_attention_flash_decoding_vsm(
            q.view(q_shape),
            cache_k,
            cache_v,
            infer_state,
            out=out,
            alloc_tensor_func=self.alloc_tensor,
        )

    def _token_decode_attention_flashattention(self, q, infer_state: FlashAttentionStateInfo, layer_weight, out=None):
        cache_k = infer_state.mem_manager.kv_buffer[self.layer_num_][:, 0 : self.tp_k_head_num_, :].reshape(
            -1, 1, self.tp_k_head_num_, self.head_dim_
        )
        cache_v = infer_state.mem_manager.kv_buffer[self.layer_num_][
            :, self.tp_k_head_num_ : self.tp_k_head_num_ + self.tp_v_head_num_, :
        ].reshape(-1, 1, self.tp_v_head_num_, self.head_dim_)
        q = q.reshape(-1, self.tp_q_head_num_, self.head_dim_)
        k_descale, v_descale = None, None  # disable quantization
        Lq = q.shape[-1]
        sm_scale = 1.0 / (Lq ** 0.5)
        o = flash_attn_with_kvcache(
            q=q,
            k_cache=cache_k,
            v_cache=cache_v,
            page_table=infer_state.page_table,
            cache_seqlens=infer_state.b_seq_len,
            cu_seqlens_q=infer_state.cu_seqlens_q,
            cu_seqlens_k_new=infer_state.cu_seqlens_k,
            max_seqlen_q=1,
            softmax_scale=sm_scale,
            causal=True,
            window_size=(-1, -1),
            softcap=0.0,
            k_descale=k_descale,
            v_descale=v_descale,
            return_softmax_lse=False,
        )
        return o

    def _token_decode_attention_flashattention_fp8(
        self, q, infer_state: FlashAttentionStateInfo, layer_weight, out=None
    ):
        cache_k = (
            (infer_state.mem_manager.kv_buffer[self.layer_num_][:, : self.tp_k_head_num_, :])
            .reshape(-1, 1, self.tp_k_head_num_, self.head_dim_)
            .view(torch.float8_e4m3fn)
        )
        cache_v = (
            (
                infer_state.mem_manager.kv_buffer[self.layer_num_][
                    :, self.tp_k_head_num_ : self.tp_k_head_num_ + self.tp_v_head_num_, :
                ]
            )
            .reshape(-1, 1, self.tp_v_head_num_, self.head_dim_)
            .view(torch.float8_e4m3fn)
        )
        q, q_scale = scaled_fp8_quant(q.view(q.shape[0] * self.tp_k_head_num_, -1), use_per_token_if_dynamic=True)
        o = flash_attn_with_kvcache(
            q=q.view(-1, self.tp_q_head_num_, self.head_dim_),
            k_cache=cache_k,
            v_cache=cache_v,
            page_table=infer_state.page_table,
            cache_seqlens=infer_state.b_seq_len,
            cu_seqlens_q=infer_state.cu_seqlens_q,
            cu_seqlens_k_new=infer_state.cu_seqlens_k,
            max_seqlen_q=1,
            causal=False,
            window_size=(-1, -1),
            softcap=0.0,
            q_descale=q_scale.view(infer_state.batch_size, self.tp_k_head_num_),
            k_descale=infer_state.k_descale[self.layer_num_],
            v_descale=infer_state.v_descale[self.layer_num_],
            return_softmax_lse=False,
        )
        return o

    def overlap_tpsp_token_forward(
        self,
        input_embdings: torch.Tensor,
        input_embdings1: torch.Tensor,
        infer_state: LlamaInferStateInfo,
        infer_state1: LlamaInferStateInfo,
        layer_weight: LlamaTransformerLayerWeight,
    ):
        input_embdings = self.tpsp_token_forward(input_embdings, infer_state, layer_weight=layer_weight)
        input_embdings1 = self.tpsp_token_forward(input_embdings1, infer_state1, layer_weight=layer_weight)
        return input_embdings, input_embdings1

    def overlap_tpsp_context_forward(
        self,
        input_embdings: torch.Tensor,
        input_embdings1: torch.Tensor,
        infer_state: LlamaInferStateInfo,
        infer_state1: LlamaInferStateInfo,
        layer_weight: LlamaTransformerLayerWeight,
    ):
        input_embdings = self.tpsp_context_forward(input_embdings, infer_state, layer_weight=layer_weight)
        input_embdings1 = self.tpsp_context_forward(input_embdings1, infer_state1, layer_weight=layer_weight)
        return input_embdings, input_embdings1
