import torch
import torch.distributed as dist
from .transformer_layer_infer_template import TransformerLayerInferTpl
from ...infer_struct import InferStateInfo
from lightllm.utils.infer_utils import mark_cost_time
from lightllm.common.basemodel.triton_kernel.destindex_copy_kv import destindex_copy_kv
from typing import Tuple


class TransformerLayerInferActivationWeightQuantTpl(TransformerLayerInferTpl):
    """
    """
    def __init__(self, layer_num, tp_rank, world_size, network_config, mode):
        super().__init__(layer_num, tp_rank, world_size, network_config, mode)
        return
    
    def _awquant_matmul_for_qkv(self, input, quant_weight_params, is_prefill, out=None, bias=None, has_act=False):
        raise Exception("need to impl")
    
    def _awquant_matmul_for_o(self, input, quant_weight_params, is_prefill, out=None, bias=None, has_act=False):
        raise Exception("need to impl")
    
    def _awquant_matmul_for_ffn_up(self, input, quant_weight_params, is_prefill, out=None, bias=None, has_act=False):
        raise Exception("need to impl")
    
    def _awquant_matmul_for_ffn_down(self, input, quant_weight_params, is_prefill, out=None, bias=None, has_act=False):
        raise Exception("need to impl")
    
    def _awquant_att_norm(self, input, infer_state:InferStateInfo, layer_weight)->torch.Tensor:
        raise Exception("need to impl")
    
    def _awquant_ffn_norm(self, input, infer_state:InferStateInfo, layer_weight)->torch.Tensor:
        raise Exception("need to impl")
    
    def _pre_cache_kv(self, infer_state:InferStateInfo, layer_weight)->Tuple[torch.Tensor, torch.Tensor]:
        '''
        Release kv buffer to save memory, since we allocate while kv projection. 
        '''
        infer_state.kv_buffer = None
        return None


