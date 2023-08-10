import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np

from lightllm.models.llama.layer_infer.transformer_layer_inference import TransformerLayerInfer
from lightllm.models.qwen.layer_weights.transformer_layer_weight import TransformerLayerWeight
from lightllm.models.llama.triton_kernel.context_flashattention_nopad import context_attention_fwd
from lightllm.models.llama.triton_kernel.rmsnorm import rmsnorm_forward
from lightllm.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd

from lightllm.models.llama.layer_infer.infer_struct import InferStateInfo
from lightllm.utils.infer_utils import mark_cost_time

torch.backends.cudnn.enabled = True


class QwenTransformerLayerInfer(TransformerLayerInfer):
    """
    """

    def __init__(self, layer_num, tp_rank, world_size, network_config, mode=""):
        super().__init__(layer_num, tp_rank, world_size, network_config, mode)
        return

    @mark_cost_time("trans context flash forward time cost")  # dont to remove this, will make performence down, did not know why
    def _context_flash_attention(self, input_embding, infer_state: InferStateInfo, layer_weight: TransformerLayerWeight):
        mem_manager = infer_state.mem_manager
        prefill_mem_index = infer_state.prefill_mem_index
        prefill_key_buffer = infer_state.prefill_key_buffer
        prefill_value_buffer = infer_state.prefill_value_buffer
        
        total_token_num = infer_state.total_token_num
        calcu_shape1 = (total_token_num, self.tp_head_num_, self.head_dim_)
        input1 = rmsnorm_forward(input_embding, weight=layer_weight.input_layernorm, eps=self.layer_norm_eps_)
        q = torch.addmm(layer_weight.q_bias_, input1.view(-1, self.embed_dim_), layer_weight.q_weight_, beta=1.0, alpha=1.0)
        rotary_emb_fwd(q.view(calcu_shape1), infer_state.position_cos, infer_state.position_sin)
        torch.addmm(layer_weight.k_bias_, input1.view(-1, self.embed_dim_), layer_weight.k_weight_, beta=1.0, alpha=1.0,
                    out=prefill_key_buffer[0:total_token_num, :, :].view(-1, self.tp_head_sum_dim_))
        rotary_emb_fwd(prefill_key_buffer[0:total_token_num, :, :], infer_state.position_cos, infer_state.position_sin)
        torch.addmm(layer_weight.v_bias_, input1.view(-1, self.embed_dim_), layer_weight.v_weight_, beta=1.0, alpha=1.0,
                    out=prefill_value_buffer[0:total_token_num, :, :].view(-1, self.tp_head_sum_dim_))
        
        input1 = None
        o_tensor = torch.empty_like(q)
        context_attention_fwd(q.view(calcu_shape1),
                              prefill_key_buffer[0:total_token_num, :, :],
                              prefill_value_buffer[0:total_token_num, :, :],
                              o_tensor.view(calcu_shape1),
                              infer_state.b_start_loc,
                              infer_state.b_seq_len,
                              infer_state.max_len_in_batch)
        self._copy_kv_to_mem_cache(prefill_key_buffer, prefill_value_buffer, prefill_mem_index, mem_manager)
        q = None
        o_tensor1 = torch.mm(o_tensor.view(-1, self.tp_head_sum_dim_), layer_weight.att_out_dense_weight_)
        o_tensor = None
        if self.world_size_ > 1:
            dist.all_reduce(o_tensor1, op=dist.ReduceOp.SUM, async_op=False)
        input_embding.add_(o_tensor1.view(total_token_num, self.embed_dim_))
        o_tensor1 = None
        return
    
    def _token_flash_attention(self, input_embding, infer_state: InferStateInfo, layer_weight: TransformerLayerWeight):
        total_token_num = infer_state.total_token_num
        batch_size = infer_state.batch_size
        calcu_shape1 = (batch_size, self.tp_head_num_, self.head_dim_)
        if infer_state.decode_is_contiguous: # int8kv always is not contiguous
            cache_k = infer_state.mem_manager.key_buffer[self.layer_num_][infer_state.decode_mem_start:infer_state.decode_mem_end, :, :]
            cache_v = infer_state.mem_manager.value_buffer[self.layer_num_][infer_state.decode_mem_start:infer_state.decode_mem_end, :, :]
        else:
            cache_k = infer_state.decode_key_buffer
            cache_v = infer_state.decode_value_buffer
            
        input1 = rmsnorm_forward(input_embding, weight=layer_weight.input_layernorm, eps=self.layer_norm_eps_)

        q = torch.addmm(layer_weight.q_bias_, input1.view(-1, self.embed_dim_), layer_weight.q_weight_, beta=1.0, alpha=1.0)
        rotary_emb_fwd(q.view(calcu_shape1), infer_state.position_cos, infer_state.position_sin)
        torch.addmm(layer_weight.k_bias_, input1.view(-1, self.embed_dim_), layer_weight.k_weight_, beta=1.0, alpha=1.0,
                    out=cache_k.view(-1, self.tp_head_sum_dim_))
        rotary_emb_fwd(cache_k,
                        infer_state.position_cos, infer_state.position_sin)
        torch.addmm(layer_weight.v_bias_, input1.view(-1, self.embed_dim_), layer_weight.v_weight_, beta=1.0, alpha=1.0,
                    out=cache_v.view(-1, self.tp_head_sum_dim_))
        
        if not infer_state.decode_is_contiguous: # int8kv always is not contiguous
            self._copy_kv_to_mem_cache(cache_k, cache_v, infer_state.decode_mem_index, infer_state.mem_manager)
             
        input1 = None
        o_tensor = self._token_decode_attention_mode(q, infer_state)
        q = None
        o_tensor1 = torch.mm(o_tensor.view(-1, self.tp_head_sum_dim_), layer_weight.att_out_dense_weight_)
        o_tensor = None
        if self.world_size_ > 1:
            dist.all_reduce(o_tensor1, op=dist.ReduceOp.SUM, async_op=False)
        input_embding.add_(o_tensor1.view(batch_size, self.embed_dim_))
        o_tensor1 = None
        return
