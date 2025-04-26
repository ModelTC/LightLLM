import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np

from lightllm.models.starcoder.layer_weights.pre_and_post_layer_weight import PreAndPostLayerWeight
from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.utils.infer_utils import mark_cost_time
from lightllm.common.basemodel import PreLayerInfer
from lightllm.models.llama.triton_kernel.embedding import embedding
from lightllm.distributed.communication_op import all_reduce


class StarcoderPreLayerInfer(PreLayerInfer):
    """ """

    def __init__(self, network_config, mode):
        super().__init__(network_config, mode)
        assert network_config["vocab_size"] % self.tp_world_size_ == 0
        self.tp_vocab_size_ = network_config["vocab_size"] // self.tp_world_size_
        self.embed_dim_ = network_config["hidden_size"]
        self.layer_norm_eps_ = network_config["layer_norm_epsilon"]
        self.vob_start_id_ = self.tp_vocab_size_ * self.tp_rank_
        self.vob_end_id_ = self.tp_vocab_size_ * (self.tp_rank_ + 1)

    def context_forward(self, input_ids, infer_state: InferStateInfo, layer_weight: PreAndPostLayerWeight):
        total_token_num = infer_state.total_token_num
        input_ids = input_ids[0:total_token_num]

        input_embdings = self.alloc_tensor(
            (input_ids.shape[0], layer_weight.wte_weight_.shape[1]), dtype=layer_weight.data_type_
        )
        embedding(input_ids, layer_weight.wte_weight_, self.vob_start_id_, self.vob_end_id_, input_embdings)
        if self.tp_world_size_ > 1:
            all_reduce(input_embdings, group=infer_state.dist_group, op=dist.ReduceOp.SUM, async_op=False)

        position_embeds = self.alloc_tensor(
            (infer_state.position_ids.shape[0], layer_weight.wpe_weight_.shape[1]), dtype=layer_weight.data_type_
        )
        embedding(
            infer_state.position_ids, layer_weight.wpe_weight_, 0, layer_weight.wpe_weight_.shape[0], position_embeds
        )

        return input_embdings.add_(position_embeds)

    def token_forward(self, input_ids, infer_state: InferStateInfo, layer_weight: PreAndPostLayerWeight):
        # import ipdb;ipdb.set_trace()
        input_embdings = self.alloc_tensor(
            (input_ids.shape[0], layer_weight.wte_weight_.shape[1]), dtype=layer_weight.data_type_
        )
        embedding(input_ids, layer_weight.wte_weight_, self.vob_start_id_, self.vob_end_id_, input_embdings)
        if self.tp_world_size_ > 1:
            all_reduce(input_embdings, group=infer_state.dist_group, op=dist.ReduceOp.SUM, async_op=False)

        position_embeds = self.alloc_tensor(
            (infer_state.position_ids.shape[0], layer_weight.wpe_weight_.shape[1]), dtype=layer_weight.data_type_
        )
        embedding(
            infer_state.position_ids, layer_weight.wpe_weight_, 0, layer_weight.wpe_weight_.shape[0], position_embeds
        )

        return input_embdings.add_(position_embeds)
