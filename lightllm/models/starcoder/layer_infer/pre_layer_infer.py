import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np

from lightllm.models.starcoder.layer_weights.pre_and_post_layer_weight import PreAndPostLayerWeight
from lightllm.models.starcoder.infer_struct import StarcoderInferStateInfo
from lightllm.utils.infer_utils import mark_cost_time
from lightllm.common.basemodel import PreLayerInfer


class StarcoderPreLayerInfer(PreLayerInfer):
    """
    """

    def __init__(self, tp_rank, world_size, network_config, mode):
        super().__init__(tp_rank, world_size, network_config, mode)
        assert (network_config["vocab_size"] % self.world_size_ == 0)
        self.tp_vocab_size_ = network_config["vocab_size"] // self.world_size_
        self.embed_dim_ = network_config["hidden_size"]
        self.layer_norm_eps_ = network_config["layer_norm_epsilon"]
        self.vob_start_id_ = self.tp_vocab_size_ * self.tp_rank_
        self.vob_end_id_ = self.tp_vocab_size_ * (self.tp_rank_ + 1)

    @mark_cost_time("pre context forward")
    def context_forward(self, input_ids, infer_state: StarcoderInferStateInfo, layer_weight: PreAndPostLayerWeight):
        total_token_num = infer_state.total_token_num
        input_ids = input_ids[0:total_token_num]

        input_mask = torch.logical_or(self.vob_start_id_ > input_ids, input_ids >= self.vob_end_id_)
        tmp_input_ids = (input_ids - self.vob_start_id_)
        tmp_input_ids[input_mask] = 0
        input_embdings = torch.embedding(layer_weight.wte_weight_, tmp_input_ids, padding_idx=-1)
        input_embdings[input_mask] = 0.0
        if self.world_size_ > 1:
            dist.all_reduce(input_embdings, op=dist.ReduceOp.SUM, async_op=False)
        position_embeds = torch.embedding(layer_weight.wpe_weight_, infer_state.position_ids, padding_idx=-1)
        
        return input_embdings + position_embeds

    def token_forward(self, input_ids, infer_state: StarcoderInferStateInfo, layer_weight: PreAndPostLayerWeight):
        # import ipdb;ipdb.set_trace()
        input_mask = torch.logical_or(self.vob_start_id_ > input_ids, input_ids >= self.vob_end_id_)
        tmp_input_ids = (input_ids - self.vob_start_id_)
        tmp_input_ids[input_mask] = 0
        input_embdings = torch.embedding(layer_weight.wte_weight_, tmp_input_ids, padding_idx=-1)
        input_embdings[input_mask] = 0.0
        if self.world_size_ > 1:
            dist.all_reduce(input_embdings, op=dist.ReduceOp.SUM, async_op=False)
        position_embeds = torch.embedding(layer_weight.wpe_weight_, infer_state.position_ids, padding_idx=-1)
        return input_embdings + position_embeds
