import torch.distributed as dist
from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.models.glm4_moe.layer_infer.transformer_layer_infer import Glm4MoeTransformerLayerInfer
from lightllm.distributed.communication_op import all_reduce
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class Glm4MoeMTPTransformerLayerInfer(Glm4MoeTransformerLayerInfer):
    """
    Transformer layer inference for GLM-4.7 MoE MTP model.

    MTP layers only run FFN (no attention), so we override the forward
    methods to skip attention computation.
    """

    def __init__(self, layer_num, network_config):
        super().__init__(layer_num, network_config)
        return

    def context_forward(self, input_embdings, infer_state: InferStateInfo, layer_weight):
        """Context (prefill) forward: only run FFN, no attention."""
        input1 = self._ffn_norm(input_embdings, infer_state, layer_weight)
        ffn_out = self._ffn(input1, infer_state, layer_weight)
        input1 = None
        if self.tp_world_size_ > 1:
            all_reduce(ffn_out, op=dist.ReduceOp.SUM, group=infer_state.dist_group, async_op=False)
        input_embdings.add_(ffn_out.view(-1, self.embed_dim_))
        return input_embdings

    def token_forward(self, input_embdings, infer_state: InferStateInfo, layer_weight):
        """Token (decode) forward: only run FFN, no attention."""
        input1 = self._ffn_norm(input_embdings, infer_state, layer_weight)
        ffn_out = self._ffn(input1, infer_state, layer_weight)
        input1 = None
        if self.tp_world_size_ > 1:
            all_reduce(ffn_out, op=dist.ReduceOp.SUM, group=infer_state.dist_group, async_op=False)
        input_embdings.add_(ffn_out.view(-1, self.embed_dim_))
        return input_embdings
