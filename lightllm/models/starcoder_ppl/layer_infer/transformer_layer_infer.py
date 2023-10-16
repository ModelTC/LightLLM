import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np

from lightllm.utils.infer_utils import mark_cost_time
from lightllm.models.starcoder.layer_infer.infer_struct import StarcoderInferStateInfo
from lightllm.models.starcoder.layer_infer.transformer_layer_infer import StarcoderTransformerLayerInfer


class StarcoderPPlTransformerLayerInfer(StarcoderTransformerLayerInfer):
    """
    """

    def __init__(self, layer_num, tp_rank,
                 world_size, network_config, mode=[]):
        super().__init__(layer_num, tp_rank, world_size, network_config, mode)
        return

    def _post_cache_kv(self, cache_k, cache_v,
                       infer_state: StarcoderInferStateInfo, layer_weight):
        mem_manager = infer_state.mem_manager
        from lightllm.models.llama_ppl.triton_kernel.quant_copy_kv import destindex_copy_quantize_kv
        if infer_state.is_prefill:
            destindex_copy_quantize_kv(cache_k,
                                       infer_state.prefill_mem_index,
                                       mem_manager.key_buffer[self.layer_num_],
                                       mem_manager.key_scale_buffer[self.layer_num_])
            destindex_copy_quantize_kv(cache_v,
                                       infer_state.prefill_mem_index,
                                       mem_manager.value_buffer[self.layer_num_],
                                       mem_manager.value_scale_buffer[self.layer_num_])
            return
        else:
            if not infer_state.decode_is_contiguous:
                destindex_copy_quantize_kv(cache_k,
                                           infer_state.decode_mem_index,
                                           mem_manager.key_buffer[self.layer_num_],
                                           mem_manager.key_scale_buffer[self.layer_num_])
                destindex_copy_quantize_kv(cache_v,
                                           infer_state.decode_mem_index,
                                           mem_manager.value_buffer[self.layer_num_],
                                           mem_manager.value_scale_buffer[self.layer_num_])
                return
        return

    def _token_attention_kernel(
            self, q, infer_state: StarcoderInferStateInfo, layer_weight) -> torch.Tensor:
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
