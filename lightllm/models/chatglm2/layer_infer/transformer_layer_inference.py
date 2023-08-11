import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np

from lightllm.models.chatglm2.layer_weights.transformer_layer_weight import TransformerLayerWeight
from lightllm.models.llama2.triton_kernel.context_flashattention_nopad import context_attention_fwd
from lightllm.models.llama2.triton_kernel.token_attention_nopad_att1 import token_att_fwd
from lightllm.models.llama2.triton_kernel.token_attention_nopad_softmax import token_softmax_fwd
from lightllm.models.llama2.triton_kernel.token_attention_nopad_reduceV import token_att_fwd2
from lightllm.models.llama.triton_kernel.rmsnorm import rmsnorm_forward
from lightllm.models.chatglm2.triton_kernel.rotary_emb import rotary_emb_fwd

from lightllm.models.chatglm2.layer_infer.infer_struct import InferStateInfo
from lightllm.utils.infer_utils import mark_cost_time
from lightllm.common.triton_kernel.destindex_copy_kv import destindex_copy_kv

torch.backends.cudnn.enabled = True


class TransformerLayerInfer:
    """
    """

    def __init__(self, layer_num, tp_rank, world_size, network_config, mode=""):
        self.layer_num_ = layer_num
        self.tp_rank_ = tp_rank
        self.world_size_ = world_size
        self.network_config_ = network_config
        self.embed_dim_ = network_config["hidden_size"]
        self.layer_norm_eps_ = network_config["layernorm_epsilon"]
        self.head_num_ = network_config["num_attention_heads"]
        self.head_dim_ = self.embed_dim_ // self.head_num_
        assert self.head_num_ % self.world_size_ == 0
        
        self.key_value_head_num_ = network_config["multi_query_group_num"]
        assert self.key_value_head_num_ % self.world_size_ == 0
        
        self.tp_head_num_ = self.head_num_ // self.world_size_
        self.tp_head_sum_dim_ = self.tp_head_num_ * self.head_dim_
        
        self.tp_kv_head_num = self.key_value_head_num_ // self.world_size_
        self.tp_kv_head_sum_dim_ = self.tp_kv_head_num * self.head_dim_
        
        self.mode = mode
        return

    @mark_cost_time("trans context flash forward time cost")  # dont to remove this, will make performence down, did not know why
    def _context_flash_attention(self, input_embding, infer_state: InferStateInfo, layer_weight: TransformerLayerWeight):
        mem_manager = infer_state.mem_manager
        prefill_mem_index = infer_state.prefill_mem_index
        prefill_key_buffer = infer_state.prefill_key_buffer
        prefill_value_buffer = infer_state.prefill_value_buffer
        
        total_token_num = infer_state.total_token_num
        calcu_shape1 = (total_token_num, self.tp_head_num_, self.head_dim_)
        input1 = rmsnorm_forward(input_embding, weight=layer_weight.input_layernorm_weight_, eps=self.layer_norm_eps_)
        q = torch.addmm(layer_weight.q_bias_, input1.view(-1, self.embed_dim_), layer_weight.q_weight_, beta=1.0, alpha=1.0)
        rotary_emb_fwd(q.view(calcu_shape1), infer_state.position_cos, infer_state.position_sin)
        torch.addmm(layer_weight.k_bias_, input1.view(-1, self.embed_dim_), layer_weight.k_weight_, beta=1.0, alpha=1.0,
                    out=prefill_key_buffer[0:total_token_num, :, :].view(-1, self.tp_kv_head_sum_dim_))
        rotary_emb_fwd(prefill_key_buffer[0:total_token_num, :, :], infer_state.position_cos, infer_state.position_sin)
        torch.addmm(layer_weight.v_bias_, input1.view(-1, self.embed_dim_), layer_weight.v_weight_, beta=1.0, alpha=1.0,
                    out=prefill_value_buffer[0:total_token_num, :, :].view(-1, self.tp_kv_head_sum_dim_))
        input1 = None
        o_tensor = torch.empty_like(q)
        context_attention_fwd(q.view(calcu_shape1),
                              prefill_key_buffer[0:total_token_num, :, :],
                              prefill_value_buffer[0:total_token_num, :, :],
                              o_tensor.view(calcu_shape1),
                              infer_state.b_start_loc,
                              infer_state.b_seq_len,
                              infer_state.max_len_in_batch)
        destindex_copy_kv(prefill_key_buffer, prefill_mem_index, mem_manager.key_buffer[self.layer_num_])
        destindex_copy_kv(prefill_value_buffer, prefill_mem_index, mem_manager.value_buffer[self.layer_num_])
        q = None
        o_tensor1 = torch.mm(o_tensor.view(-1, self.tp_head_sum_dim_), layer_weight.att_out_dense_weight_)
        o_tensor = None
        if self.world_size_ > 1:
            dist.all_reduce(o_tensor1, op=dist.ReduceOp.SUM, async_op=False)
        input_embding.add_(o_tensor1.view(total_token_num, self.embed_dim_))
        return

    def swiglu(self, x):
        x = torch.chunk(x, 2, dim=-1)
        return torch.nn.functional.silu(x[0]) * x[1]

    @mark_cost_time("trans context ffn forward time cost")
    def _context_ffn(self, input_embdings, infer_state: InferStateInfo, layer_weight: TransformerLayerWeight):
        total_token_num = infer_state.total_token_num
        batch_size = infer_state.batch_size
        input1 = rmsnorm_forward(input_embdings,
                                 weight=layer_weight.post_attention_layernorm_weight_,
                                 eps=self.layer_norm_eps_)

        ffn1_out = torch.mm(input1.view(-1, self.embed_dim_), layer_weight.ffn_1_weight_)
        act_out = self.swiglu(ffn1_out)
        ffn1_out = None
        ffn2_out = torch.mm(act_out, layer_weight.ffn_2_weight_)
        if self.world_size_ > 1:
            dist.all_reduce(ffn2_out, op=dist.ReduceOp.SUM, async_op=False)
        input_embdings.add_(ffn2_out.view(total_token_num, self.embed_dim_))
        return 

    def context_forward(self, input_embdings, infer_state: InferStateInfo, layer_weight: TransformerLayerWeight):
        self._context_flash_attention(input_embdings,
                                      infer_state,
                                      layer_weight=layer_weight)
        self._context_ffn(input_embdings, infer_state, layer_weight)
        return input_embdings

    def _token_flash_attention(self, input_embding, infer_state: InferStateInfo, layer_weight: TransformerLayerWeight):
        total_token_num = infer_state.total_token_num
        batch_size = infer_state.batch_size
        calcu_shape1 = (batch_size, self.tp_head_num_, self.head_dim_)
        if infer_state.decode_is_contiguous:
            cache_k = infer_state.mem_manager.key_buffer[self.layer_num_][infer_state.decode_mem_start:infer_state.decode_mem_end, :, :]
            cache_v = infer_state.mem_manager.value_buffer[self.layer_num_][infer_state.decode_mem_start:infer_state.decode_mem_end, :, :]
        else:
            cache_k = infer_state.decode_key_buffer
            cache_v = infer_state.decode_value_buffer
            
        input1 = rmsnorm_forward(input_embding, weight=layer_weight.input_layernorm_weight_, eps=self.layer_norm_eps_)

        q = torch.addmm(layer_weight.q_bias_, input1.view(-1, self.embed_dim_), layer_weight.q_weight_, beta=1.0, alpha=1.0)

        rotary_emb_fwd(q.view(calcu_shape1), infer_state.position_cos, infer_state.position_sin)
        torch.addmm(layer_weight.k_bias_, input1.view(-1, self.embed_dim_), layer_weight.k_weight_, beta=1.0, alpha=1.0,
                    out=cache_k.view(-1, self.tp_kv_head_sum_dim_))
        rotary_emb_fwd(cache_k,
                        infer_state.position_cos, infer_state.position_sin)
        torch.addmm(layer_weight.v_bias_, input1.view(-1, self.embed_dim_), layer_weight.v_weight_, beta=1.0, alpha=1.0,
                    out=cache_v.view(-1, self.tp_kv_head_sum_dim_))

        if not infer_state.decode_is_contiguous:
            destindex_copy_kv(cache_k, infer_state.decode_mem_index, infer_state.mem_manager.key_buffer[self.layer_num_])
            destindex_copy_kv(cache_v, infer_state.decode_mem_index, infer_state.mem_manager.value_buffer[self.layer_num_])
             
        att_m_tensor = torch.empty((self.tp_head_num_, total_token_num), dtype=q.dtype, device="cuda")

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

        q = None
        o_tensor1 = torch.mm(o_tensor.view(-1, self.tp_head_sum_dim_), layer_weight.att_out_dense_weight_)
        o_tensor = None
        if self.world_size_ > 1:
            dist.all_reduce(o_tensor1, op=dist.ReduceOp.SUM, async_op=False)
        input_embding.add_(o_tensor1.view(batch_size, self.embed_dim_))
        return
        

    def _token_ffn(self, input_embdings, infer_state: InferStateInfo, layer_weight: TransformerLayerWeight):
        total_token_num = infer_state.total_token_num
        batch_size = infer_state.batch_size
        input1 = rmsnorm_forward(input_embdings,
                                 weight=layer_weight.post_attention_layernorm_weight_,
                                 eps=self.layer_norm_eps_)

        ffn1_out = torch.mm(input1.view(-1, self.embed_dim_), layer_weight.ffn_1_weight_)
        act_out = self.swiglu(ffn1_out)
        ffn1_out = None
        ffn2_out = torch.mm(act_out, layer_weight.ffn_2_weight_)
        if self.world_size_ > 1:
            dist.all_reduce(ffn2_out, op=dist.ReduceOp.SUM, async_op=False)
        input_embdings.add_(ffn2_out.view(batch_size, self.embed_dim_))
        return


    def token_forward(self, input_embdings, infer_state: InferStateInfo, layer_weight: TransformerLayerWeight):
        self._token_flash_attention(input_embdings,
                            infer_state,
                            layer_weight=layer_weight)
        self._token_ffn(input_embdings, infer_state, layer_weight)
        return input_embdings
