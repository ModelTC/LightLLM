import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np
from typing import Tuple

from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.models.llama2.layer_weights.transformer_layer_weight import  LlamaTransformerLayerWeight
from lightllm.models.llama.infer_struct import LlamaInferStateInfo

class LlamaPPlTransformerLayerInfer(LlamaTransformerLayerInfer):
    """
    """

    def __init__(self, layer_num, tp_rank, world_size, network_config, mode=[]):
        super().__init__(layer_num, tp_rank, world_size, network_config, mode)
        return

    def _ffn(self, input, infer_state:LlamaInferStateInfo, layer_weight:LlamaTransformerLayerWeight)->torch.Tensor:
        gate_out = torch.mm(input.view(-1, self.embed_dim_), layer_weight.gate_proj)
        torch.nn.functional.silu(gate_out, inplace=True)
        up_out = torch.mm(input.view(-1, self.embed_dim_), layer_weight.up_proj)
        input = None
        ffn1_out = gate_out * up_out
        gate_out, up_out = None, None
        ffn2_out = torch.mm(ffn1_out, layer_weight.down_proj)
        ffn1_out = None
        return ffn2_out
    
    def _copy_kv_to_mem_cache(self, key_buffer, value_buffer, mem_index, mem_manager):
        from lightllm.models.llama_ppl.triton_kernel.quant_copy_kv import destindex_copy_quantize_kv
        destindex_copy_quantize_kv(key_buffer,
                                    mem_index,
                                    mem_manager.key_buffer[self.layer_num_],
                                    mem_manager.key_scale_buffer[self.layer_num_])
        destindex_copy_quantize_kv(value_buffer,
                                    mem_index,
                                    mem_manager.value_buffer[self.layer_num_],
                                    mem_manager.value_scale_buffer[self.layer_num_])

    def _token_decode_attention_int8kv(self, q, infer_state: LlamaInferStateInfo):
        total_token_num = infer_state.total_token_num
        batch_size = infer_state.batch_size
        calcu_shape1 = (batch_size, self.tp_q_head_num_, self.head_dim_)
        att_m_tensor = torch.empty((self.tp_q_head_num_, total_token_num), dtype=q.dtype, device="cuda")
        o_tensor = torch.empty_like(q)
        
        from lightllm_ppl_kernel import group8_int8kv_decode_attention
        # group_int8kv_decode_attention(at::Tensor o, at::Tensor q, at::Tensor k, at::Tensor k_s,  at::Tensor v,  at::Tensor v_s, at::Tensor b_loc, at::Tensor b_seq_len, int max_len_in_batch)
        group8_int8kv_decode_attention(o_tensor.view(calcu_shape1),
                                                          q.view(calcu_shape1),
                                                          infer_state.mem_manager.key_buffer[self.layer_num_],
                                                          infer_state.mem_manager.key_scale_buffer[self.layer_num_],
                                                          infer_state.mem_manager.value_buffer[self.layer_num_],
                                                          infer_state.mem_manager.value_scale_buffer[self.layer_num_],
                                                          infer_state.b_loc,
                                                          infer_state.b_seq_len,
                                                          infer_state.max_len_in_batch)
           
        return o_tensor
    
    def _token_decode_attention_mode(self, q, infer_state: LlamaInferStateInfo):
        return self._token_decode_attention_int8kv(q, infer_state)