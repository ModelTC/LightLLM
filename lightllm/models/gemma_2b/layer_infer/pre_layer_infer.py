import torch
import torch.distributed as dist
import numpy as np

from lightllm.models.gemma_2b.layer_weights.pre_and_post_layer_weight import Gemma_2bPreAndPostLayerWeight
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.common.basemodel import PreLayerInferTpl
from lightllm.distributed.communication_op import all_reduce


class Gemma_2bPreLayerInfer(PreLayerInferTpl):
    """ """

    def __init__(self, network_config, mode):
        super().__init__(network_config, mode)
        tp_vob_ids = np.linspace(0, network_config["vocab_size"], self.tp_world_size_ + 1, dtype=np.int64)
        self.vob_start_id_, self.vob_end_id_ = int(tp_vob_ids[self.tp_rank_]), int(tp_vob_ids[self.tp_rank_ + 1])
        self.normfactor = network_config["hidden_size"] ** 0.5
        return

    def _norm(self, input, infer_state, layer_weight: Gemma_2bPreAndPostLayerWeight) -> torch.Tensor:
        return input * self.normfactor

    def context_forward(self, input_ids, infer_state: LlamaInferStateInfo, layer_weight: Gemma_2bPreAndPostLayerWeight):
        input_embdings = layer_weight.wte_weight_.embedding(
            input_ids=input_ids,
            alloc_func=self.alloc_tensor,
        )
        if self.tp_world_size_ > 1:
            all_reduce(input_embdings, group=infer_state.dist_group, op=dist.ReduceOp.SUM, async_op=False)
        input_embdings = self._norm(input_embdings, infer_state, layer_weight)
        return input_embdings

    def token_forward(self, input_ids, infer_state: LlamaInferStateInfo, layer_weight: Gemma_2bPreAndPostLayerWeight):
        input_embdings = layer_weight.wte_weight_.embedding(
            input_ids=input_ids,
            alloc_func=self.alloc_tensor,
        )
        if self.tp_world_size_ > 1:
            all_reduce(input_embdings, group=infer_state.dist_group, op=dist.ReduceOp.SUM, async_op=False)
        input_embdings = self._norm(input_embdings, infer_state, layer_weight)
        return input_embdings
