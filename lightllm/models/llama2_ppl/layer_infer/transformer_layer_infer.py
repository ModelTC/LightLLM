import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np
from typing import Tuple

from lightllm.models.llama2.layer_infer.transformer_layer_infer import Llama2TransformerLayerInfer
from lightllm.models.llama.infer_struct import LlamaInferStateInfo

class Llama2PPlTransformerLayerInfer(Llama2TransformerLayerInfer):
    """
    """

    def __init__(self, layer_num, tp_rank, world_size, network_config, mode=[]):
        super().__init__(layer_num, tp_rank, world_size, network_config, mode)
        return
    
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

    def _token_attention_kernel(self, q, infer_state:LlamaInferStateInfo, layer_weight)->torch.Tensor:
        batch_size = infer_state.batch_size
        calcu_shape1 = (batch_size, self.tp_q_head_num_, self.head_dim_)
        o_tensor = torch.empty_like(q)
        
        from lightllm_ppl_kernel import group8_int8kv_decode_attention
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