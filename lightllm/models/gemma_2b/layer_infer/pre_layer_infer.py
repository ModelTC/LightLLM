import torch
import torch.distributed as dist
import numpy as np

from lightllm.common.basemodel.splitfuse_infer_struct import SplitFuseInferStateInfo
from lightllm.models.gemma_2b.layer_weights.pre_and_post_layer_weight import Gemma_2bPreAndPostLayerWeight
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.common.basemodel import PreLayerInferTpl
from lightllm.utils.infer_utils import mark_cost_time


class Gemma_2bPreLayerInfer(PreLayerInferTpl):
    """ """

    def __init__(self, tp_rank, world_size, network_config, mode):
        super().__init__(tp_rank, world_size, network_config, mode)
        tp_vob_ids = np.linspace(0, network_config["vocab_size"], self.world_size_ + 1, dtype=np.int64)
        self.vob_start_id_, self.vob_end_id_ = int(tp_vob_ids[self.tp_rank_]), int(tp_vob_ids[self.tp_rank_ + 1])
        self.normfactor = network_config["hidden_size"]**0.5
        return

    def _norm(self, input, infer_state, layer_weight : Gemma_2bPreAndPostLayerWeight) -> torch.Tensor:
        return input * self.normfactor

    @mark_cost_time("pre context forward")
    def context_forward(self, input_ids, infer_state: LlamaInferStateInfo, layer_weight: Gemma_2bPreAndPostLayerWeight):
        input_mask = torch.logical_or(self.vob_start_id_ > input_ids, input_ids >= self.vob_end_id_)
        tmp_input_ids = input_ids - self.vob_start_id_
        tmp_input_ids[input_mask] = 0
        input_embdings = torch.embedding(layer_weight.wte_weight_, tmp_input_ids, padding_idx=-1)
        input_embdings[input_mask] = 0.0
        if self.world_size_ > 1:
            dist.all_reduce(input_embdings, op=dist.ReduceOp.SUM, async_op=False)
        input_embdings = self._norm(input_embdings, infer_state, layer_weight)
        return input_embdings

    def token_forward(self, input_ids, infer_state: LlamaInferStateInfo, layer_weight: Gemma_2bPreAndPostLayerWeight):
        input_mask = torch.logical_or(self.vob_start_id_ > input_ids, input_ids >= self.vob_end_id_)
        tmp_input_ids = input_ids - self.vob_start_id_
        tmp_input_ids[input_mask] = 0
        input_embdings = torch.embedding(layer_weight.wte_weight_, tmp_input_ids, padding_idx=-1)
        input_embdings[input_mask] = 0.0
        if self.world_size_ > 1:
            dist.all_reduce(input_embdings, op=dist.ReduceOp.SUM, async_op=False)
        input_embdings = self._norm(input_embdings, infer_state, layer_weight)
        return input_embdings

    # @mark_cost_time("splitfuse forward")
    def splitfuse_forward(
        self, input_ids, infer_state: SplitFuseInferStateInfo, layer_weight: Gemma_2bPreAndPostLayerWeight
    ):
        return self.token_forward(input_ids, infer_state, layer_weight)
