from typing import Tuple

import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np

from lightllm.common.basemodel.triton_kernel.dequantize_gemm_int4 import matmul_dequantize_int4_s1, \
    matmul_dequantize_int4_gptq
from lightllm.common.basemodel.triton_kernel.dequantize_gemm_int8 import matmul_dequantize_int8
from lightllm.models.starcoder_quantized.layer_weights.transformer_layer_weight import \
    StarcoderTransformerLayerWeightQuantized
from lightllm.utils.infer_utils import mark_cost_time
from lightllm.models.starcoder.layer_infer.infer_struct import StarcoderInferStateInfo
from lightllm.models.starcoder.layer_infer.transformer_layer_infer import StarcoderTransformerLayerInfer
from lightllm.models.starcoder.layer_weights.transformer_layer_weight import StarcoderTransformerLayerWeight

from lightllm.common.basemodel.triton_kernel.destindex_copy_kv import destindex_copy_kv, destindex_copy_quantize_kv
from lightllm.models.bloom.triton_kernel.layernorm import layernorm_forward
from lightllm.models.llama2.triton_kernel.context_flashattention_nopad import context_attention_fwd
from lightllm.models.llama2.triton_kernel.token_attention_nopad_att1 import token_att_fwd
from lightllm.models.llama2.triton_kernel.token_attention_nopad_softmax import token_softmax_fwd
from lightllm.models.llama2.triton_kernel.token_attention_nopad_reduceV import token_att_fwd2


class StarcoderTransformerLayerInferINT8(StarcoderTransformerLayerInfer):
    """
    """
    def __init__(self, layer_num, tp_rank, world_size, network_config, mode=[]):
        super().__init__(layer_num, tp_rank, world_size, network_config, mode)
        self.tp_k_head_num_ = 1
        self.tp_v_head_num_ = 1
        self.head_dim = network_config["hidden_size"] // network_config["num_attention_heads"]
        return

    def _get_qkv(self, input, infer_state: StarcoderInferStateInfo, layer_weight: StarcoderTransformerLayerWeightQuantized,
                 matmul_int8_func) -> torch.Tensor:
        qkv_output = layer_weight.qkv_fused_bias + matmul_int8_func(input.view(-1, self.embed_dim_), layer_weight.qkv_fused_weight, layer_weight.qkv_fused_weight_scale)
        q = qkv_output[:, : -2 * self.head_dim]
        k = qkv_output[:, -2 * self.head_dim: -self.head_dim]
        v = qkv_output[:, -self.head_dim:]
        cache_k = k.view(-1, self.tp_k_head_num_, self.head_dim_)
        cache_v = v.view(-1, self.tp_v_head_num_, self.head_dim_)
        return q, cache_k, cache_v

    def _get_qkv_context(self, input, infer_state: StarcoderInferStateInfo, layer_weight: StarcoderTransformerLayerWeightQuantized) -> torch.Tensor:
        return self._get_qkv(input, infer_state, layer_weight, matmul_dequantize_int8)

    def _get_qkv_decode(self, input, infer_state: StarcoderInferStateInfo, layer_weight: StarcoderTransformerLayerWeightQuantized) -> torch.Tensor:
        return self._get_qkv(input, infer_state, layer_weight, matmul_dequantize_int8)

    def _get_o_context(self, input, infer_state: StarcoderInferStateInfo, layer_weight: StarcoderTransformerLayerWeightQuantized) -> torch.Tensor:
        o_output = layer_weight.o_bias_ / self.world_size_ + matmul_dequantize_int8(input.view(-1, self.tp_o_head_num_ * self.head_dim_),
                                          layer_weight.o_weight_, layer_weight.o_weight_scale_)
        return o_output

    def _get_o_decode(self, input, infer_state: StarcoderInferStateInfo, layer_weight: StarcoderTransformerLayerWeightQuantized) -> torch.Tensor:
        o_output = layer_weight.o_bias_ / self.world_size_ + matmul_dequantize_int8(input.view(-1, self.tp_o_head_num_ * self.head_dim_),
                                          layer_weight.o_weight_, layer_weight.o_weight_scale_)
        return o_output

    def _ffn(self, input, infer_state:StarcoderInferStateInfo, layer_weight: StarcoderTransformerLayerWeightQuantized, matmul_int8_func)->torch.Tensor:
        ffn1_out = layer_weight.ffn_1_bias_ + matmul_int8_func(input.view(-1, self.embed_dim_), layer_weight.ffn_1_weight_, layer_weight.ffn_1_weight_scale)
        input = None
        gelu_out = torch.nn.functional.gelu(ffn1_out, approximate='tanh')
        ffn1_out = None
        ffn2_out = layer_weight.ffn_2_bias_ / self.world_size_ + matmul_int8_func(gelu_out, layer_weight.ffn_2_weight_, layer_weight.ffn_2_weight_scale)
        gelu_out = None
        return ffn2_out

    def _ffn_context(self, input, infer_state: StarcoderInferStateInfo, layer_weight: StarcoderTransformerLayerWeightQuantized) -> torch.Tensor:
        return self._ffn(input, infer_state, layer_weight, matmul_dequantize_int8)

    def _ffn_decode(self, input, infer_state: StarcoderInferStateInfo, layer_weight: StarcoderTransformerLayerWeightQuantized) -> torch.Tensor:
        return self._ffn(input, infer_state, layer_weight, matmul_dequantize_int8)

    @mark_cost_time("trans context flash forward time cost")
    def _context_attention(self, input_embding, infer_state: StarcoderInferStateInfo, layer_weight):
        input1 = self._att_norm(input_embding, infer_state, layer_weight)
        self._pre_cache_kv(infer_state, layer_weight)
        q, cache_k, cache_v = self._get_qkv_context(input1, infer_state, layer_weight)
        input1 = None
        self._post_cache_kv(cache_k, cache_v, infer_state, layer_weight)
        o = self._context_attention_kernel(q, cache_k, cache_v, infer_state, layer_weight)
        q = None
        o = self._get_o_context(o, infer_state, layer_weight)
        if self.world_size_ > 1:
            dist.all_reduce(o, op=dist.ReduceOp.SUM, async_op=False)
        input_embding.add_(o.view(-1, self.embed_dim_))

    @mark_cost_time("trans context ffn forward time cost")  # dont to remove this, will make performence down, did not know why
    def _context_ffn(self, input_embdings, infer_state: StarcoderInferStateInfo, layer_weight):
        input1 = self._ffn_norm(input_embdings, infer_state, layer_weight)
        ffn_out = self._ffn_context(input1, infer_state, layer_weight)
        input1 = None
        if self.world_size_ > 1:
            dist.all_reduce(ffn_out, op=dist.ReduceOp.SUM, async_op=False)
        input_embdings.add_(ffn_out.view(-1, self.embed_dim_))

    def _token_attention(self, input_embding, infer_state: StarcoderInferStateInfo, layer_weight):
        input1 = self._att_norm(input_embding, infer_state, layer_weight)
        self._pre_cache_kv(infer_state, layer_weight)
        q, cache_k, cache_v = self._get_qkv_decode(input1, infer_state, layer_weight)
        input1 = None
        self._post_cache_kv(cache_k, cache_v, infer_state, layer_weight)
        o = self._token_attention_kernel(q, infer_state, layer_weight)
        q = None
        o = self._get_o_decode(o, infer_state, layer_weight)
        if self.world_size_ > 1:
            dist.all_reduce(o, op=dist.ReduceOp.SUM, async_op=False)
        input_embding.add_(o.view(-1, self.embed_dim_))

    def _token_ffn(self, input_embdings, infer_state: StarcoderInferStateInfo, layer_weight):
        input1 = self._ffn_norm(input_embdings, infer_state, layer_weight)
        ffn_out = self._ffn_decode(input1, infer_state, layer_weight)
        input1 = None
        if self.world_size_ > 1:
            dist.all_reduce(ffn_out, op=dist.ReduceOp.SUM, async_op=False)
        input_embdings.add_(ffn_out.view(-1, self.embed_dim_))

    # Pre / post cache kv for fused weight.
    def _pre_cache_kv(self, infer_state: StarcoderInferStateInfo, layer_weight)->Tuple[torch.Tensor, torch.Tensor]:
        '''
        Release kv buffer to save memory, since we allocate while kv projection.
        '''
        if infer_state.is_prefill:
            infer_state.prefill_key_buffer = None
            infer_state.prefill_value_buffer = None
        else:
            infer_state.decode_key_buffer = None
            infer_state.decode_value_buffer = None

    def _post_cache_kv(self, cache_k, cache_v, infer_state: StarcoderInferStateInfo, layer_weight: StarcoderTransformerLayerWeightQuantized):
        mem_manager = infer_state.mem_manager
        if infer_state.is_prefill:
            destindex_copy_kv(cache_k, infer_state.prefill_mem_index, mem_manager.key_buffer[self.layer_num_])
            destindex_copy_kv(cache_v, infer_state.prefill_mem_index, mem_manager.value_buffer[self.layer_num_])
        else:
            destindex_copy_kv(cache_k, infer_state.decode_mem_index, mem_manager.key_buffer[self.layer_num_])
            destindex_copy_kv(cache_v, infer_state.decode_mem_index, mem_manager.value_buffer[self.layer_num_])


class StarcoderTransformerLayerInferINT4(StarcoderTransformerLayerInfer):
    """
    """
    def __init__(self, layer_num, tp_rank, world_size, network_config, mode=[], group_size=128):
        super().__init__(layer_num, tp_rank, world_size, network_config, mode)
        self.tp_k_head_num_ = 1
        self.tp_v_head_num_ = 1
        self.head_dim = network_config["hidden_size"] // network_config["num_attention_heads"]
        self.q_group_size = group_size
        return

    def _get_qkv(self, input, infer_state: StarcoderInferStateInfo, layer_weight: StarcoderTransformerLayerWeightQuantized,
                 matmul_int4_func) -> torch.Tensor:
        qkv_output = layer_weight.qkv_fused_bias + matmul_int4_func(input.view(-1, self.embed_dim_), layer_weight.qkv_fused_weight, layer_weight.qkv_fused_weight_scale,
                                                                    layer_weight.qkv_fused_weight_zp, self.q_group_size)
        q = qkv_output[:, : -2 * self.head_dim]
        k = qkv_output[:, -2 * self.head_dim: -self.head_dim]
        v = qkv_output[:, -self.head_dim:]
        cache_k = k.view(-1, self.tp_k_head_num_, self.head_dim_)
        cache_v = v.view(-1, self.tp_v_head_num_, self.head_dim_)
        return q, cache_k, cache_v

    def _get_qkv_context(self, input, infer_state: StarcoderInferStateInfo, layer_weight: StarcoderTransformerLayerWeightQuantized) -> torch.Tensor:
        return self._get_qkv(input, infer_state, layer_weight, matmul_dequantize_int4_s1)

    def _get_qkv_decode(self, input, infer_state: StarcoderInferStateInfo, layer_weight: StarcoderTransformerLayerWeightQuantized) -> torch.Tensor:
        return self._get_qkv(input, infer_state, layer_weight, matmul_dequantize_int4_gptq)

    def _get_o_context(self, input, infer_state: StarcoderInferStateInfo, layer_weight: StarcoderTransformerLayerWeightQuantized) -> torch.Tensor:
        o_output = layer_weight.o_bias_ / self.world_size_ + matmul_dequantize_int4_s1(input.view(-1, self.tp_o_head_num_ * self.head_dim_),
                                          layer_weight.o_weight_, layer_weight.o_weight_scale_, layer_weight.o_weight_zp_, self.q_group_size)
        return o_output

    def _get_o_decode(self, input, infer_state: StarcoderInferStateInfo, layer_weight: StarcoderTransformerLayerWeightQuantized) -> torch.Tensor:
        o_output = layer_weight.o_bias_ / self.world_size_ + matmul_dequantize_int4_gptq(input.view(-1, self.tp_o_head_num_ * self.head_dim_),
                                          layer_weight.o_weight_, layer_weight.o_weight_scale_, layer_weight.o_weight_zp_, self.q_group_size)
        return o_output

    def _ffn(self, input, infer_state:StarcoderInferStateInfo, layer_weight: StarcoderTransformerLayerWeightQuantized, matmul_int4_func)->torch.Tensor:
        ffn1_out = layer_weight.ffn_1_bias_ + matmul_int4_func(input.view(-1, self.embed_dim_), layer_weight.ffn_1_weight_, layer_weight.ffn_1_weight_scale, layer_weight.ffn_1_weight_zp, self.q_group_size)
        input = None
        gelu_out = torch.nn.functional.gelu(ffn1_out, approximate='tanh')
        ffn1_out = None
        ffn2_out = layer_weight.ffn_2_bias_ / self.world_size_ + matmul_int4_func(gelu_out, layer_weight.ffn_2_weight_, layer_weight.ffn_2_weight_scale, layer_weight.ffn_2_weight_zp, self.q_group_size)
        gelu_out = None
        return ffn2_out

    def _ffn_context(self, input, infer_state: StarcoderInferStateInfo, layer_weight: StarcoderTransformerLayerWeightQuantized) -> torch.Tensor:
        return self._ffn(input, infer_state, layer_weight, matmul_dequantize_int4_s1)

    def _ffn_decode(self, input, infer_state: StarcoderInferStateInfo, layer_weight: StarcoderTransformerLayerWeightQuantized) -> torch.Tensor:
        return self._ffn(input, infer_state, layer_weight, matmul_dequantize_int4_gptq)

    @mark_cost_time("trans context flash forward time cost")
    def _context_attention(self, input_embding, infer_state: StarcoderInferStateInfo, layer_weight):
        input1 = self._att_norm(input_embding, infer_state, layer_weight)
        self._pre_cache_kv(infer_state, layer_weight)
        q, cache_k, cache_v = self._get_qkv_context(input1, infer_state, layer_weight)
        input1 = None
        self._post_cache_kv(cache_k, cache_v, infer_state, layer_weight)
        o = self._context_attention_kernel(q, cache_k, cache_v, infer_state, layer_weight)
        q = None
        o = self._get_o_context(o, infer_state, layer_weight)
        if self.world_size_ > 1:
            dist.all_reduce(o, op=dist.ReduceOp.SUM, async_op=False)
        input_embding.add_(o.view(-1, self.embed_dim_))

    @mark_cost_time("trans context ffn forward time cost")  # dont to remove this, will make performence down, did not know why
    def _context_ffn(self, input_embdings, infer_state: StarcoderInferStateInfo, layer_weight):
        input1 = self._ffn_norm(input_embdings, infer_state, layer_weight)
        ffn_out = self._ffn_context(input1, infer_state, layer_weight)
        input1 = None
        if self.world_size_ > 1:
            dist.all_reduce(ffn_out, op=dist.ReduceOp.SUM, async_op=False)
        input_embdings.add_(ffn_out.view(-1, self.embed_dim_))

    def _token_attention(self, input_embding, infer_state: StarcoderInferStateInfo, layer_weight):
        input1 = self._att_norm(input_embding, infer_state, layer_weight)
        self._pre_cache_kv(infer_state, layer_weight)
        q, cache_k, cache_v = self._get_qkv_decode(input1, infer_state, layer_weight)
        input1 = None
        self._post_cache_kv(cache_k, cache_v, infer_state, layer_weight)
        o = self._token_attention_kernel(q, infer_state, layer_weight)
        q = None
        o = self._get_o_decode(o, infer_state, layer_weight)
        if self.world_size_ > 1:
            dist.all_reduce(o, op=dist.ReduceOp.SUM, async_op=False)
        input_embding.add_(o.view(-1, self.embed_dim_))

    def _token_ffn(self, input_embdings, infer_state: StarcoderInferStateInfo, layer_weight):
        input1 = self._ffn_norm(input_embdings, infer_state, layer_weight)
        ffn_out = self._ffn_decode(input1, infer_state, layer_weight)
        input1 = None
        if self.world_size_ > 1:
            dist.all_reduce(ffn_out, op=dist.ReduceOp.SUM, async_op=False)
        input_embdings.add_(ffn_out.view(-1, self.embed_dim_))

    # Pre / post cache kv for fused weight.
    def _pre_cache_kv(self, infer_state: StarcoderInferStateInfo, layer_weight)->Tuple[torch.Tensor, torch.Tensor]:
        '''
        Release kv buffer to save memory, since we allocate while kv projection.
        '''
        if infer_state.is_prefill:
            infer_state.prefill_key_buffer = None
            infer_state.prefill_value_buffer = None
        else:
            infer_state.decode_key_buffer = None
            infer_state.decode_value_buffer = None

    def _post_cache_kv(self, cache_k, cache_v, infer_state: StarcoderInferStateInfo, layer_weight: StarcoderTransformerLayerWeightQuantized):
        mem_manager = infer_state.mem_manager
        if infer_state.is_prefill:
            destindex_copy_kv(cache_k, infer_state.prefill_mem_index, mem_manager.key_buffer[self.layer_num_])
            destindex_copy_kv(cache_v, infer_state.prefill_mem_index, mem_manager.value_buffer[self.layer_num_])
        else:
            destindex_copy_kv(cache_k, infer_state.decode_mem_index, mem_manager.key_buffer[self.layer_num_])
            destindex_copy_kv(cache_v, infer_state.decode_mem_index, mem_manager.value_buffer[self.layer_num_])

    def _token_decode_attention_normal(self, q, infer_state: StarcoderInferStateInfo):
        '''
        We override token decode attention normal using triton kernel for 2.0.0 version,
        since there is bug in kernel for 2.1.0 version when using int4 weight.
        '''
        total_token_num = infer_state.total_token_num
        batch_size = infer_state.batch_size
        calcu_shape1 = (batch_size, self.tp_q_head_num_, self.head_dim_)
        att_m_tensor = torch.empty((self.tp_q_head_num_, total_token_num), dtype=q.dtype, device="cuda")

        token_att_fwd(q.view(calcu_shape1),
                      infer_state.mem_manager.key_buffer[self.layer_num_],
                      att_m_tensor,
                      infer_state.b_loc,
                      infer_state.b_start_loc,
                      infer_state.b_seq_len,
                      infer_state.max_len_in_batch)

        prob = torch.empty_like(att_m_tensor)
        token_softmax_fwd(att_m_tensor, infer_state.b_start_loc, infer_state.b_seq_len, prob, infer_state.max_len_in_batch)
        att_m_tensor = None

        o_tensor = torch.empty_like(q)

        token_att_fwd2(prob,
                       infer_state.mem_manager.value_buffer[self.layer_num_],
                       o_tensor.view(calcu_shape1),
                       infer_state.b_loc,
                       infer_state.b_start_loc,
                       infer_state.b_seq_len,
                       infer_state.max_len_in_batch)
        prob = None
        return o_tensor