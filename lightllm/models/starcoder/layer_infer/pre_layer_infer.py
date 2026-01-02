import torch.distributed as dist

from lightllm.models.starcoder.layer_weights.pre_and_post_layer_weight import StarcoderPreAndPostLayerWeight
from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.common.basemodel import PreLayerInfer
from lightllm.distributed.communication_op import all_reduce


class StarcoderPreLayerInfer(PreLayerInfer):
    """ """

    def __init__(self, network_config, mode):
        super().__init__(network_config, mode)
        self.layer_norm_eps_ = network_config["layer_norm_epsilon"]

    def context_forward(self, input_ids, infer_state: InferStateInfo, layer_weight: StarcoderPreAndPostLayerWeight):
        input_embdings = layer_weight.wte_weight_.embedding(input_ids=input_ids, alloc_func=self.alloc_tensor)
        if self.tp_world_size_ > 1:
            all_reduce(input_embdings, group=infer_state.dist_group, op=dist.ReduceOp.SUM, async_op=False)

        position_embeds = layer_weight.wpe_weight_.embedding(
            input_ids=infer_state.position_ids,
            alloc_func=self.alloc_tensor,
        )

        return input_embdings.add_(position_embeds)

    def token_forward(self, input_ids, infer_state: InferStateInfo, layer_weight: StarcoderPreAndPostLayerWeight):
        input_embdings = layer_weight.wte_weight_.embedding(input_ids=input_ids, alloc_func=self.alloc_tensor)
        if self.tp_world_size_ > 1:
            all_reduce(input_embdings, group=infer_state.dist_group, op=dist.ReduceOp.SUM, async_op=False)

        position_embeds = layer_weight.wpe_weight_.embedding(
            input_ids=infer_state.position_ids,
            alloc_func=self.alloc_tensor,
        )
        return input_embdings.add_(position_embeds)
