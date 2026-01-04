import torch
import torch.distributed as dist
from lightllm.models.llama.layer_weights.pre_and_post_layer_weight import LlamaPreAndPostLayerWeight
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.common.basemodel import PreLayerInferTpl
from lightllm.distributed.communication_op import all_reduce
from lightllm.utils.envs_utils import get_env_start_args


class LlamaPreLayerInfer(PreLayerInferTpl):
    """ """

    def __init__(self, network_config, mode):
        super().__init__(network_config, mode)
        return

    def context_forward(self, input_ids, infer_state: LlamaInferStateInfo, layer_weight: LlamaPreAndPostLayerWeight):
        input_embdings = layer_weight.wte_weight_.embedding(input_ids=input_ids, alloc_func=self.alloc_tensor)
        if self.tp_world_size_ > 1:
            all_reduce(input_embdings, op=dist.ReduceOp.SUM, group=infer_state.dist_group, async_op=False)
        return input_embdings

    def token_forward(self, input_ids, infer_state: LlamaInferStateInfo, layer_weight: LlamaPreAndPostLayerWeight):
        input_embdings = layer_weight.wte_weight_.embedding(input_ids=input_ids, alloc_func=self.alloc_tensor)
        if self.tp_world_size_ > 1:
            all_reduce(input_embdings, op=dist.ReduceOp.SUM, group=infer_state.dist_group, async_op=False)
        return input_embdings

    def tpsp_context_forward(
        self, input_ids, infer_state: LlamaInferStateInfo, layer_weight: LlamaPreAndPostLayerWeight
    ):
        if get_env_start_args().enable_dp_prefill_balance:
            input_ids = infer_state.prefill_dp_balance(input_ids=input_ids)

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

    def overlap_tpsp_context_forward(
        self,
        input_ids: torch.Tensor,
        input_ids1: torch.Tensor,
        infer_state: LlamaInferStateInfo,
        infer_state1: LlamaInferStateInfo,
        layer_weight: LlamaPreAndPostLayerWeight,
    ):
        if get_env_start_args().enable_dp_prefill_balance:
            input_ids = infer_state.prefill_dp_balance(input_ids=input_ids)

        input_embdings = self.context_forward(input_ids=input_ids, infer_state=infer_state, layer_weight=layer_weight)
        from lightllm.common.basemodel.triton_kernel.sp_pad_copy import sp_pad_copy

        padded_input_embdings = sp_pad_copy(input_embdings, sp_rank_id=self.tp_rank_, sp_world_size=self.tp_world_size_)

        if get_env_start_args().enable_dp_prefill_balance:
            input_ids1 = infer_state1.prefill_dp_balance(input_ids=input_ids1)

        input_embdings1 = self.context_forward(
            input_ids=input_ids1, infer_state=infer_state1, layer_weight=layer_weight
        )
        from lightllm.common.basemodel.triton_kernel.sp_pad_copy import sp_pad_copy

        padded_input_embdings1 = sp_pad_copy(
            input_embdings1, sp_rank_id=self.tp_rank_, sp_world_size=self.tp_world_size_
        )

        return padded_input_embdings, padded_input_embdings1
