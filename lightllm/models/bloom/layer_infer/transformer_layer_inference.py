import time
import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np

from lightllm.common.basemodel import TransformerLayerInfer
from lightllm.models.bloom.layer_weights.transformer_layer_weight import BloomTransformerLayerWeight
from lightllm.models.bloom.triton_kernel.context_flashattention_nopad import context_attention_fwd
from lightllm.models.bloom.triton_kernel.token_flashattention_nopad import token_attention_fwd
from lightllm.models.bloom.triton_kernel.layernorm import layernorm_forward
from lightllm.common.basemodel import InferStateInfo
from lightllm.utils.infer_utils import mark_cost_time
from lightllm.common.basemodel.triton_kernel.destindex_copy_kv import destindex_copy_kv


class BloomTransformerLayerInfer(TransformerLayerInfer):
    """
    """

    def __init__(self, layer_num, tp_rank, world_size, network_config, mode):
        super().__init__(layer_num, tp_rank, world_size, network_config, mode)
        self.embed_dim_ = network_config["n_embed"]
        self.layer_norm_eps_ = network_config["layer_norm_epsilon"]
        self.head_num_ = network_config["num_attention_heads"]
        self.head_dim_ = self.embed_dim_ // self.head_num_
        assert self.head_num_ % self.world_size_ == 0
        self.tp_head_num_ = self.head_num_ // self.world_size_
        self.tp_head_sum_dim_ = self.tp_head_num_ * self.head_dim_
        return

    @mark_cost_time("trans context flash forward time cost")  # dont to remove this, will make performence down, did not know why
    def _context_flash_attention(self, input_embding, infer_state: InferStateInfo, layer_weight: BloomTransformerLayerWeight):
        mem_manager = infer_state.mem_manager
        prefill_mem_index = infer_state.prefill_mem_index
        prefill_key_buffer = infer_state.prefill_key_buffer
        prefill_value_buffer = infer_state.prefill_value_buffer
        total_token_num = infer_state.total_token_num
        input1 = layernorm_forward(
            input_embding,
            weight=layer_weight.input_layernorm_weight_,
            bias=layer_weight.input_layernorm_bias_,
            eps=self.layer_norm_eps_)
        q = torch.addmm(layer_weight.q_bias_, input1.view(-1, self.embed_dim_), layer_weight.q_weight_, beta=1.0, alpha=1.0)
        torch.addmm(layer_weight.k_bias_, input1.view(-1, self.embed_dim_), layer_weight.k_weight_, beta=1.0,
                    alpha=1.0, out=prefill_key_buffer.view(-1, self.tp_head_sum_dim_))
        torch.addmm(layer_weight.v_bias_, input1.view(-1, self.embed_dim_), layer_weight.v_weight_, beta=1.0,
                    alpha=1.0, out=prefill_value_buffer.view(-1, self.tp_head_sum_dim_))
        input1 = None
        calcu_shape1 = (total_token_num, self.tp_head_num_, self.head_dim_)
        o_tensor = torch.empty_like(q)
        context_attention_fwd(q.view(calcu_shape1),
                              prefill_key_buffer,
                              prefill_value_buffer,
                              o_tensor.view(calcu_shape1),
                              layer_weight.tp_alibi,
                              infer_state.b_start_loc,
                              infer_state.b_seq_len,
                              infer_state.max_len_in_batch)
        destindex_copy_kv(prefill_key_buffer, prefill_mem_index, mem_manager.key_buffer[self.layer_num_])
        destindex_copy_kv(prefill_value_buffer, prefill_mem_index, mem_manager.value_buffer[self.layer_num_])
        q = None
        o_tensor1 = torch.addmm(layer_weight.att_out_dense_bias_,
                                o_tensor.view(-1,
                                              self.tp_head_sum_dim_),
                                layer_weight.att_out_dense_weight_,
                                beta=1.0 / self.world_size_)  # 这个地方用的bias偏执，是本身模型的 1 / 8, 这样可以在后面dist reduce的时候不在需要加偏执操作了。
        o_tensor = None
        if self.world_size_ > 1:
            dist.all_reduce(o_tensor1, op=dist.ReduceOp.SUM, async_op=False)
        input_embding.add_(o_tensor1.view(total_token_num, self.embed_dim_))
        o_tensor1 = None
        return

    @mark_cost_time("trans context ffn forward time cost")  # dont to remove this, will make performence down, did not know why
    def _context_ffn(self, input_embdings, infer_state: InferStateInfo, layer_weight: BloomTransformerLayerWeight):
        total_token_num = infer_state.total_token_num
        input1 = layernorm_forward(input_embdings,
                                   weight=layer_weight.post_attention_layernorm_weight_,
                                   bias=layer_weight.post_attention_layernorm_bias_,
                                   eps=self.layer_norm_eps_)
        ffn1_out = torch.addmm(layer_weight.ffn_1_bias_, input1.view(-1, self.embed_dim_), layer_weight.ffn_1_weight_)
        input1 = None
        gelu_out = torch.nn.functional.gelu(ffn1_out, approximate='tanh')
        ffn1_out = None

        ffn2_out = torch.addmm(layer_weight.ffn_2_bias_, gelu_out, layer_weight.ffn_2_weight_, beta=1.0 / self.world_size_)
        gelu_out = None
        if self.world_size_ > 1:
            dist.all_reduce(ffn2_out, op=dist.ReduceOp.SUM, async_op=False)
        input_embdings.add_(ffn2_out.view(total_token_num, self.embed_dim_))
        ffn2_out = None
        return

    def context_forward(self, input_embdings, infer_state: InferStateInfo, layer_weight: BloomTransformerLayerWeight):
        self._context_flash_attention(input_embdings,
                                      infer_state,
                                      layer_weight=layer_weight)
        self._context_ffn(input_embdings, infer_state, layer_weight)
        return input_embdings

    def _token_flash_attention(self, input_embding, infer_state: InferStateInfo, layer_weight: BloomTransformerLayerWeight):

        total_token_num = infer_state.total_token_num
        batch_size = infer_state.batch_size
        if infer_state.decode_is_contiguous:
            cache_k = infer_state.mem_manager.key_buffer[self.layer_num_][infer_state.decode_mem_start:infer_state.decode_mem_end, :, :]
            cache_v = infer_state.mem_manager.value_buffer[self.layer_num_][infer_state.decode_mem_start:infer_state.decode_mem_end, :, :]
        else:
            cache_k = infer_state.decode_key_buffer
            cache_v = infer_state.decode_value_buffer
        
        input1 = layernorm_forward(
            input_embding,
            weight=layer_weight.input_layernorm_weight_,
            bias=layer_weight.input_layernorm_bias_,
            eps=self.layer_norm_eps_)
        q = torch.addmm(layer_weight.q_bias_, input1.view(-1, self.embed_dim_), layer_weight.q_weight_, beta=1.0, alpha=1.0)
        torch.addmm(layer_weight.k_bias_, input1.view(-1, self.embed_dim_), layer_weight.k_weight_, beta=1.0, alpha=1.0,
                    out=cache_k.view(-1, self.tp_head_sum_dim_))
        torch.addmm(layer_weight.v_bias_, input1.view(-1, self.embed_dim_), layer_weight.v_weight_, beta=1.0, alpha=1.0,
                    out=cache_v.view(-1, self.tp_head_sum_dim_))
        if not infer_state.decode_is_contiguous:
            destindex_copy_kv(cache_k, infer_state.decode_mem_index, infer_state.mem_manager.key_buffer[self.layer_num_])
            destindex_copy_kv(cache_v, infer_state.decode_mem_index, infer_state.mem_manager.value_buffer[self.layer_num_])
        input1 = None
        calcu_shape1 = (batch_size, self.tp_head_num_, self.head_dim_)

        o_tensor = torch.empty_like(q)
        token_attention_fwd(q.view(calcu_shape1),
                            infer_state.mem_manager.key_buffer[self.layer_num_],
                            infer_state.mem_manager.value_buffer[self.layer_num_],
                            o_tensor.view(calcu_shape1),
                            layer_weight.tp_alibi,
                            infer_state.b_loc,
                            infer_state.b_start_loc,
                            infer_state.b_seq_len,
                            infer_state.max_len_in_batch)

        q = None
        o_tensor1 = torch.addmm(layer_weight.att_out_dense_bias_,
                                o_tensor.view(-1,
                                              self.tp_head_sum_dim_),
                                layer_weight.att_out_dense_weight_,
                                beta=1.0 / self.world_size_)  # 这个地方用的bias偏执，是本身模型的 1 / 8, 这样可以在后面dist reduce的时候不在需要加偏执操作了。
        o_tensor = None
        if self.world_size_ > 1:
            dist.all_reduce(o_tensor1, op=dist.ReduceOp.SUM, async_op=False)
        input_embding.add_(o_tensor1.view(batch_size, self.embed_dim_))
        o_tensor1 = None
        return

    def _token_ffn(self, input_embdings, infer_state: InferStateInfo, layer_weight: BloomTransformerLayerWeight):
        total_token_num = infer_state.total_token_num
        batch_size = infer_state.batch_size
        input1 = layernorm_forward(input_embdings,
                                   weight=layer_weight.post_attention_layernorm_weight_,
                                   bias=layer_weight.post_attention_layernorm_bias_,
                                   eps=self.layer_norm_eps_)
        ffn1_out = torch.addmm(layer_weight.ffn_1_bias_, input1.view(-1, self.embed_dim_), layer_weight.ffn_1_weight_)
        input1 = None
        gelu_out = torch.nn.functional.gelu(ffn1_out, approximate='tanh')
        ffn1_out = None

        ffn2_out = torch.addmm(layer_weight.ffn_2_bias_, gelu_out, layer_weight.ffn_2_weight_, beta=1.0 / self.world_size_)
        gelu_out = None
        if self.world_size_ > 1:
            dist.all_reduce(ffn2_out, op=dist.ReduceOp.SUM, async_op=False)
        input_embdings.add_(ffn2_out.view(batch_size, self.embed_dim_))
        ffn2_out = None
        return

    def token_forward(self, input_embdings, infer_state: InferStateInfo, layer_weight: BloomTransformerLayerWeight):
        self._token_flash_attention(input_embdings,
                                    infer_state,
                                    layer_weight=layer_weight)
        self._token_ffn(input_embdings, infer_state, layer_weight)
        return input_embdings
