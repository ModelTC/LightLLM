import torch
import torch.distributed as dist
from lightllm.models.llama.layer_weights.pre_and_post_layer_weight import LlamaPreAndPostLayerWeight
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.common.basemodel import PreLayerInferTpl
from lightllm.distributed.communication_op import all_reduce
from lightllm.utils.envs_utils import get_env_start_args


class LlamaPreLayerInfer(PreLayerInferTpl):
    """ """

    def __init__(self, network_config):
        super().__init__(network_config)
        return

    def context_forward(self, input_ids, infer_state: LlamaInferStateInfo, layer_weight: LlamaPreAndPostLayerWeight):
        if get_env_start_args().enable_dp_prefill_balance:
            input_ids = infer_state.prefill_dp_balance(input_ids=input_ids)

        input_embdings = layer_weight.wte_weight_(input_ids=input_ids, alloc_func=self.alloc_tensor)
        if self.tp_world_size_ > 1:
            all_reduce(input_embdings, op=dist.ReduceOp.SUM, group=infer_state.dist_group, async_op=False)
        input_embdings = self._tpsp_sp_split(input=input_embdings, infer_state=infer_state)
        return input_embdings

    def token_forward(self, input_ids, infer_state: LlamaInferStateInfo, layer_weight: LlamaPreAndPostLayerWeight):
        input_embdings = layer_weight.wte_weight_(input_ids=input_ids, alloc_func=self.alloc_tensor)
        if self.tp_world_size_ > 1:
            all_reduce(input_embdings, op=dist.ReduceOp.SUM, group=infer_state.dist_group, async_op=False)
        input_embdings = self._tpsp_sp_split(input=input_embdings, infer_state=infer_state)
        return input_embdings

    def overlap_tpsp_token_forward(
        self,
        input_ids: torch.Tensor,
        input_ids1: torch.Tensor,
        infer_state: LlamaInferStateInfo,
        infer_state1: LlamaInferStateInfo,
        layer_weight: LlamaPreAndPostLayerWeight,
    ):

        input_embdings = self.token_forward(input_ids=input_ids, infer_state=infer_state, layer_weight=layer_weight)
        input_embdings1 = self.token_forward(input_ids=input_ids1, infer_state=infer_state1, layer_weight=layer_weight)

        return input_embdings, input_embdings1

    def overlap_tpsp_context_forward(
        self,
        input_ids: torch.Tensor,
        input_ids1: torch.Tensor,
        infer_state: LlamaInferStateInfo,
        infer_state1: LlamaInferStateInfo,
        layer_weight: LlamaPreAndPostLayerWeight,
    ):

        input_embdings = self.context_forward(input_ids=input_ids, infer_state=infer_state, layer_weight=layer_weight)
        input_embdings1 = self.context_forward(
            input_ids=input_ids1, infer_state=infer_state1, layer_weight=layer_weight
        )

        return input_embdings, input_embdings1
