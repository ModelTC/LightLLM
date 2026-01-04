import torch.functional as F
import torch.distributed as dist
import numpy as np
from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.models.mistral.layer_infer.transformer_layer_infer import MistralTransformerLayerInfer
from lightllm.distributed.communication_op import all_reduce
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class MistralMTPTransformerLayerInfer(MistralTransformerLayerInfer):
    def __init__(self, layer_num, network_config, mode=[]):
        super().__init__(layer_num, network_config, mode)
        return

    def context_forward(self, input_embdings, infer_state: InferStateInfo, layer_weight):
        input1 = self._ffn_norm(input_embdings, infer_state, layer_weight)
        ffn_out = self._ffn(input1, infer_state, layer_weight)
        input1 = None
        if self.tp_world_size_ > 1:
            all_reduce(ffn_out, op=dist.ReduceOp.SUM, group=infer_state.dist_group, async_op=False)
        input_embdings.add_(ffn_out.view(-1, self.embed_dim_))
        return input_embdings

    def token_forward(self, input_embdings, infer_state: InferStateInfo, layer_weight):
        input1 = self._ffn_norm(input_embdings, infer_state, layer_weight)
        ffn_out = self._ffn(input1, infer_state, layer_weight)
        input1 = None
        if self.tp_world_size_ > 1:
            all_reduce(ffn_out, op=dist.ReduceOp.SUM, group=infer_state.dist_group, async_op=False)
        input_embdings.add_(ffn_out.view(-1, self.embed_dim_))
        return input_embdings
