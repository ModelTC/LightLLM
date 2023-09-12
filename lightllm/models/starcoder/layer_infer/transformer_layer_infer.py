import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np

from lightllm.utils.infer_utils import mark_cost_time
from lightllm.models.starcoder.layer_infer.infer_struct import StarcoderInferStateInfo
from lightllm.models.bloom.layer_infer.transformer_layer_infer import BloomTransformerLayerInfer
from lightllm.models.starcoder.layer_weights.transformer_layer_weight import StarcoderTransformerLayerWeight

from lightllm.common.basemodel.triton_kernel.destindex_copy_kv import destindex_copy_kv, destindex_copy_quantize_kv
from lightllm.models.bloom.triton_kernel.layernorm import layernorm_forward
from lightllm.models.llama2.triton_kernel.context_flashattention_nopad import context_attention_fwd
from lightllm.models.llama2.triton_kernel.token_attention_nopad_att1 import token_att_fwd
from lightllm.models.llama2.triton_kernel.token_attention_nopad_softmax import token_softmax_fwd
from lightllm.models.llama2.triton_kernel.token_attention_nopad_reduceV import token_att_fwd2


class StarcoderTransformerLayerInfer(BloomTransformerLayerInfer):
    """
    """
    def __init__(self, layer_num, tp_rank, world_size, network_config, mode=[]):
        super().__init__(layer_num, tp_rank, world_size, network_config, mode)
        self.tp_k_head_num_ = 1
        self.tp_v_head_num_ = 1
        return

    def _context_attention_kernel(self, q, k, v, infer_state:StarcoderInferStateInfo, layer_weight: StarcoderTransformerLayerWeight)->torch.Tensor:
        o_tensor = torch.empty_like(q)
        context_attention_fwd(q.view(-1, self.tp_q_head_num_, self.head_dim_),
                              k.view(-1, self.tp_k_head_num_, self.head_dim_),
                              v.view(-1, self.tp_v_head_num_, self.head_dim_),
                              o_tensor.view(-1, self.tp_q_head_num_, self.head_dim_),
                              infer_state.b_start_loc,
                              infer_state.b_seq_len,
                              infer_state.max_len_in_batch)
        return o_tensor

    def _token_attention_kernel(self, q, infer_state:StarcoderInferStateInfo, layer_weight: StarcoderTransformerLayerWeight)->torch.Tensor:
        
        calcu_shape1 = (infer_state.batch_size, self.tp_q_head_num_, self.head_dim_)
        att_m_tensor = torch.empty((self.tp_q_head_num_, infer_state.total_token_num), dtype=q.dtype, device="cuda")
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
        return o_tensor
