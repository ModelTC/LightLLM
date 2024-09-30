import torch
import torch.distributed as dist
import numpy as np

from lightllm.common.basemodel.splitfuse_infer_struct import SplitFuseInferStateInfo
from lightllm.models.llama.layer_weights.pre_and_post_layer_weight import LlamaPreAndPostLayerWeight
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.common.basemodel import PreLayerInferTpl
from lightllm.utils.infer_utils import mark_cost_time


class LlamaPreLayerInfer(PreLayerInferTpl):
    """ """

    def __init__(self, tp_rank, world_size, network_config, mode):
        super().__init__(tp_rank, world_size, network_config, mode)
        tp_vob_ids = np.linspace(0, network_config["vocab_size"], self.world_size_ + 1, dtype=np.int64)
        self.vob_start_id_, self.vob_end_id_ = int(tp_vob_ids[self.tp_rank_]), int(tp_vob_ids[self.tp_rank_ + 1])
        return

    def context_forward(self, input_ids, infer_state: LlamaInferStateInfo, layer_weight: LlamaPreAndPostLayerWeight):
        input_mask = self.alloc_tensor(input_ids.shape, data_type=torch.bool)
        torch.logical_or(self.vob_start_id_ > input_ids, input_ids >= self.vob_end_id_, out=input_mask)
        tmp_input_ids = self.alloc_tensor(input_ids.shape, data_type=input_ids.dtype)
        torch.sub(input_ids, self.vob_start_id_, out=tmp_input_ids)
        tmp_input_ids[input_mask] = 0
        # to do 将 embedding 操作替换为可以 out 参数的算子，可以自己申请tensor进行传入。
        input_embdings = torch.embedding(layer_weight.wte_weight_, tmp_input_ids, padding_idx=-1)
        input_embdings[input_mask] = 0.0
        if self.world_size_ > 1:
            dist.all_reduce(input_embdings, op=dist.ReduceOp.SUM, async_op=False)
        return input_embdings

    def token_forward(self, input_ids, infer_state: LlamaInferStateInfo, layer_weight: LlamaPreAndPostLayerWeight):
        input_mask = self.alloc_tensor(input_ids.shape, data_type=torch.bool)
        torch.logical_or(self.vob_start_id_ > input_ids, input_ids >= self.vob_end_id_, out=input_mask)
        tmp_input_ids = self.alloc_tensor(input_ids.shape, data_type=input_ids.dtype)
        torch.sub(input_ids, self.vob_start_id_, out=tmp_input_ids)
        tmp_input_ids[input_mask] = 0
        # to do 将 embedding 操作替换为可以 out 参数的算子，可以自己申请tensor进行传入。
        input_embdings = torch.embedding(layer_weight.wte_weight_, tmp_input_ids, padding_idx=-1)
        input_embdings[input_mask] = 0.0
        if self.world_size_ > 1:
            dist.all_reduce(input_embdings, op=dist.ReduceOp.SUM, async_op=False)
        return input_embdings

    def splitfuse_forward(
        self, input_ids, infer_state: SplitFuseInferStateInfo, layer_weight: LlamaPreAndPostLayerWeight
    ):
        return self.token_forward(input_ids, infer_state, layer_weight)
