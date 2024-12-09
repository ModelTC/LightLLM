import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np
from typing import Tuple
from functools import partial
import triton

from lightllm.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight
from lightllm.models.llama.triton_kernel.context_flashattention_nopad import (
    context_attention_fwd,
    context_attention_fwd_no_prompt_cache,
    context_attention_fwd_ppl_int8kv,
)
from lightllm.models.llama.triton_kernel.token_attention_nopad_att1 import token_att_fwd, token_att_fwd_int8k
from lightllm.models.llama.triton_kernel.token_attention_nopad_softmax import token_softmax_fwd
from lightllm.models.llama.triton_kernel.token_attention_nopad_reduceV import token_att_fwd2, token_att_fwd2_int8v
from lightllm.models.llama.triton_kernel.rmsnorm import rmsnorm_forward
from lightllm.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.models.llama.triton_kernel.silu_and_mul import silu_and_mul_fwd

from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.models.llama.splitfuse_infer_struct import SplitFuseInferStateInfo
from lightllm.common.basemodel.triton_kernel.destindex_copy_kv import destindex_copy_kv, destindex_copy_quantize_kv
from lightllm.common.basemodel import TransformerLayerInferTpl
from lightllm.models.llama.triton_kernel.ppl_quant_copy_kv import destindex_copy_dequantize_kv


class LlamaTransformerLayerInfer(TransformerLayerInferTpl):
    """ """

    def __init__(self, layer_num, tp_rank, world_size, network_config, mode=[]):
        super().__init__(layer_num, tp_rank, world_size, network_config, mode)
        self.eps_ = network_config["rms_norm_eps"]
        self.tp_q_head_num_ = network_config["num_attention_heads"] // self.world_size_
        self.tp_k_head_num_ = network_config["num_key_value_heads"] // self.world_size_
        self.tp_v_head_num_ = network_config["num_key_value_heads"] // self.world_size_
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
        else:
            self._token_attention_kernel = partial(LlamaTransformerLayerInfer._token_decode_attention_normal, self)
            self._copy_kv_to_mem_cache = partial(LlamaTransformerLayerInfer._copy_kv_to_mem_cache_normal, self)

        # bind splitfuse attention
        if "triton_int8kv" in self.mode:
            self._splitfuse_attention_kernel = partial(
                LlamaTransformerLayerInfer._splitfuse_attention_kernel_int8kv, self
            )
        else:
            self._splitfuse_attention_kernel = partial(LlamaTransformerLayerInfer._splitfuse_attention_kernel, self)
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
        self, input, cache_kv, infer_state: LlamaInferStateInfo, layer_weight: LlamaTransformerLayerWeight
    ) -> torch.Tensor:
        q = layer_weight.q_proj.mm(input)
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

    def _context_attention_kernel(
        self, q, kv, infer_state: LlamaInferStateInfo, layer_weight, out=None
    ) -> torch.Tensor:
        o_tensor = self.alloc_tensor(q.shape, q.dtype) if out is None else out
        if infer_state.use_dynamic_prompt_cache:
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
        else:
            context_attention_fwd_no_prompt_cache(
                q.view(-1, self.tp_q_head_num_, self.head_dim_),
                kv[:, 0 : self.tp_k_head_num_, :],
                kv[:, self.tp_k_head_num_ : self.tp_k_head_num_ + self.tp_v_head_num_, :],
                o_tensor.view(-1, self.tp_q_head_num_, self.head_dim_),
                infer_state.b_start_loc,
                infer_state.b_seq_len,
                infer_state.max_len_in_batch,
            )

        return o_tensor

    def _context_attention_kernel_ppl_int8kv(
        self, q, kv, infer_state: LlamaInferStateInfo, layer_weight, out=None
    ) -> torch.Tensor:
        o_tensor = self.alloc_tensor(q.shape, q.dtype) if out is None else out
        if infer_state.use_dynamic_prompt_cache:
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
        else:
            context_attention_fwd_no_prompt_cache(
                q.view(-1, self.tp_q_head_num_, self.head_dim_),
                kv[:, 0 : self.tp_k_head_num_, :],
                kv[:, self.tp_k_head_num_ : self.tp_k_head_num_ + self.tp_v_head_num_, :],
                o_tensor.view(-1, self.tp_q_head_num_, self.head_dim_),
                infer_state.b_start_loc,
                infer_state.b_seq_len,
                infer_state.max_len_in_batch,
            )

        return o_tensor

    def _splitfuse_attention_kernel(
        self, q, infer_state: SplitFuseInferStateInfo, layer_weight, out=None
    ) -> torch.Tensor:
        o_tensor = self.alloc_tensor(q.shape, q.dtype) if out is None else out
        infer_state.start_event.record(torch.cuda.default_stream())
        if infer_state.decode_req_num > 0:
            self._token_attention_kernel(
                q[0 : infer_state.decode_req_num, :],
                infer_state.inner_decode_infer_status,
                layer_weight,
                out=o_tensor[0 : infer_state.decode_req_num, :],
            )
        calcu_shape1 = (-1, self.tp_q_head_num_, self.head_dim_)
        if infer_state.prefill_req_num > 0:
            infer_state.parrall_stream.wait_event(infer_state.start_event)
            # infer_state.start_event.wait(infer_state.parrall_stream)
            with torch.cuda.stream(infer_state.parrall_stream):
                # assert torch.cuda.current_stream().cuda_stream == infer_state.parrall_stream.cuda_stream
                self._context_attention_kernel(
                    q[infer_state.decode_req_num :, :].view(calcu_shape1),
                    infer_state.mem_manager.kv_buffer[self.layer_num_],
                    infer_state.inner_prefill_infer_status,
                    layer_weight,
                    out=o_tensor[infer_state.decode_req_num :, :].view(calcu_shape1),
                )
            infer_state.end_event.record(infer_state.parrall_stream)
            torch.cuda.default_stream().wait_event(infer_state.end_event)
            # infer_state.event.wait(torch.cuda.default_stream())
            # assert torch.cuda.current_stream().cuda_stream == torch.cuda.default_stream().cuda_stream
            # assert torch.cuda.default_stream().cuda_stream != infer_state.parrall_stream.cuda_stream
        return o_tensor

    def _splitfuse_attention_kernel_int8kv(
        self, q, infer_state: SplitFuseInferStateInfo, layer_weight, out=None
    ) -> torch.Tensor:
        o_tensor = self.alloc_tensor(q.shape, q.dtype) if out is None else out
        infer_state.start_event.record(torch.cuda.default_stream())
        if infer_state.decode_req_num > 0:
            self._token_attention_kernel(
                q[0 : infer_state.decode_req_num, :],
                infer_state.inner_decode_infer_status,
                layer_weight,
                out=o_tensor[0 : infer_state.decode_req_num, :],
            )
        calcu_shape1 = (-1, self.tp_q_head_num_, self.head_dim_)
        if infer_state.prefill_req_num > 0:
            infer_state.parrall_stream.wait_event(infer_state.start_event)
            with torch.cuda.stream(infer_state.parrall_stream):
                self._context_attention_kernel_ppl_int8kv(
                    q[infer_state.decode_req_num :, :].view(calcu_shape1),
                    infer_state.mem_manager.kv_buffer[self.layer_num_],
                    infer_state.inner_prefill_infer_status,
                    layer_weight,
                    out=o_tensor[infer_state.decode_req_num :, :].view(calcu_shape1),
                )
            infer_state.end_event.record(infer_state.parrall_stream)
            torch.cuda.default_stream().wait_event(infer_state.end_event)
        return o_tensor

    def _get_o(
        self, input, infer_state: LlamaInferStateInfo, layer_weight: LlamaTransformerLayerWeight
    ) -> torch.Tensor:
        input = input.view(-1, self.tp_o_head_num_ * self.head_dim_)
        o_tensor = layer_weight.o_proj.mm(input)
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

    def _copy_kv_to_mem_cache_int8kv(self, buffer, mem_index, mem_manager):
        destindex_copy_quantize_kv(
            buffer, mem_index, mem_manager.kv_buffer[self.layer_num_], mem_manager.scale_buffer[self.layer_num_]
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
            infer_state.other_kv_index,
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

        from lightllm_ppl_kernel import group8_int8kv_decode_attention

        # group_int8kv_decode_attention(at::Tensor o, at::Tensor q, at::Tensor k, at::Tensor k_s,  at::Tensor v,
        # at::Tensor v_s, at::Tensor b_loc, at::Tensor b_seq_len, int max_len_in_batch)
        group8_int8kv_decode_attention(
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
