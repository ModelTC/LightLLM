import torch
import torch.distributed as dist
from lightllm.common.basemodel import PreLayerInferTpl
from lightllm.common.basemodel import InferStateInfo
from lightllm.models.bloom.layer_weights.pre_and_post_layer_weight import BloomPreAndPostLayerWeight
from lightllm.distributed.communication_op import all_reduce


class BloomPreLayerInfer(PreLayerInferTpl):
    """ """

    def __init__(self, network_config):
        super().__init__(network_config)
        self.eps_ = network_config["layer_norm_epsilon"]
        return

    def _norm(self, input, infer_state, layer_weight: BloomPreAndPostLayerWeight) -> torch.Tensor:
        return layer_weight.pre_norm_weight_.layernorm_forward(input=input, eps=self.eps_, alloc_func=self.alloc_tensor)

    def context_forward(self, input_ids, infer_state: InferStateInfo, layer_weight: BloomPreAndPostLayerWeight):
        input_embdings = layer_weight.wte_weight_.embedding(input_ids=input_ids, alloc_func=self.alloc_tensor)
        if self.tp_world_size_ > 1:
            all_reduce(input_embdings, group=infer_state.dist_group, op=dist.ReduceOp.SUM, async_op=False)
        input_embdings = self._norm(input_embdings, infer_state, layer_weight)
        return input_embdings

    def token_forward(self, input_ids, infer_state: InferStateInfo, layer_weight: BloomPreAndPostLayerWeight):
        input_embdings = layer_weight.wte_weight_.embedding(input_ids=input_ids, alloc_func=self.alloc_tensor)
        if self.tp_world_size_ > 1:
            all_reduce(input_embdings, group=infer_state.dist_group, op=dist.ReduceOp.SUM, async_op=False)
        input_embdings = self._norm(input_embdings, infer_state, layer_weight)
        return input_embdings
