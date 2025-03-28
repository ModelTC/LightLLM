import os
import torch
import torch.distributed as dist
import numpy as np

from lightllm.models.llama.layer_weights.pre_and_post_layer_weight import LlamaPreAndPostLayerWeight
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.common.basemodel import PreLayerInferTpl
from lightllm.utils.infer_utils import mark_cost_time
from lightllm.models.llama.triton_kernel.embedding import embedding
from lightllm.distributed.communication_op import all_reduce


class LlamaPreLayerInfer(PreLayerInferTpl):
    """ """

    def __init__(self, network_config, mode):
        super().__init__(network_config, mode)
        tp_vob_ids = np.linspace(0, network_config["vocab_size"], self.tp_world_size_ + 1, dtype=np.int64)
        self.vob_start_id_, self.vob_end_id_ = int(tp_vob_ids[self.tp_rank_]), int(tp_vob_ids[self.tp_rank_ + 1])

        return

    def context_forward(self, input_ids, infer_state: LlamaInferStateInfo, layer_weight: LlamaPreAndPostLayerWeight):
        input_embdings = self.alloc_tensor(
            (input_ids.shape[0], layer_weight.wte_weight_.shape[1]), dtype=layer_weight.data_type_
        )
        embedding(input_ids, layer_weight.wte_weight_, self.vob_start_id_, self.vob_end_id_, input_embdings)
        if self.tp_world_size_ > 1:
            all_reduce(input_embdings, op=dist.ReduceOp.SUM, group=infer_state.dist_group, async_op=False)
        return input_embdings

    def token_forward(self, input_ids, infer_state: LlamaInferStateInfo, layer_weight: LlamaPreAndPostLayerWeight):
        input_embdings = self.alloc_tensor(
            (input_ids.shape[0], layer_weight.wte_weight_.shape[1]), dtype=layer_weight.data_type_
        )
        embedding(input_ids, layer_weight.wte_weight_, self.vob_start_id_, self.vob_end_id_, input_embdings)
        if self.tp_world_size_ > 1:
            all_reduce(input_embdings, op=dist.ReduceOp.SUM, group=infer_state.dist_group, async_op=False)
        return input_embdings

    def tpsp_context_forward(
        self, input_ids, infer_state: LlamaInferStateInfo, layer_weight: LlamaPreAndPostLayerWeight
    ):
        input_embdings = self.context_forward(input_ids=input_ids, infer_state=infer_state, layer_weight=layer_weight)
        from lightllm.common.basemodel.triton_kernel.sp_pad_copy import sp_pad_copy

        padded_input_embdings = sp_pad_copy(input_embdings, sp_rank_id=self.tp_rank_, sp_world_size=self.tp_world_size_)
        return padded_input_embdings

    def tpsp_token_forward(self, input_ids, infer_state: LlamaInferStateInfo, layer_weight: LlamaPreAndPostLayerWeight):
        input_embdings = self.token_forward(input_ids=input_ids, infer_state=infer_state, layer_weight=layer_weight)
        from lightllm.common.basemodel.triton_kernel.sp_pad_copy import sp_pad_copy

        padded_input_embdings = sp_pad_copy(input_embdings, sp_rank_id=self.tp_rank_, sp_world_size=self.tp_world_size_)
        return padded_input_embdings

    def overlap_tpsp_token_forward(
        self,
        input_ids: torch.Tensor,
        input_ids1: torch.Tensor,
        infer_state: LlamaInferStateInfo,
        infer_state1: LlamaInferStateInfo,
        layer_weight: LlamaPreAndPostLayerWeight,
    ):

        input_embdings = self.token_forward(input_ids=input_ids, infer_state=infer_state, layer_weight=layer_weight)
        from lightllm.common.basemodel.triton_kernel.sp_pad_copy import sp_pad_copy

        padded_input_embdings = sp_pad_copy(input_embdings, sp_rank_id=self.tp_rank_, sp_world_size=self.tp_world_size_)

        input_embdings1 = self.token_forward(input_ids=input_ids1, infer_state=infer_state1, layer_weight=layer_weight)
        from lightllm.common.basemodel.triton_kernel.sp_pad_copy import sp_pad_copy

        padded_input_embdings1 = sp_pad_copy(
            input_embdings1, sp_rank_id=self.tp_rank_, sp_world_size=self.tp_world_size_
        )

        return padded_input_embdings, padded_input_embdings1
