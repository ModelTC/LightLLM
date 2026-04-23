import torch
import torch.distributed as dist
from typing import Dict, Iterable, Literal, Tuple, Union, List
from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.common.basemodel.layer_weights.base_layer_weight import BaseLayerWeight
from lightllm.utils.dist_utils import get_current_rank_in_dp, get_dp_world_size
from .cache_tensor_manager import g_cache_manager
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.distributed.communication_op import all_gather_into_tensor, reduce_scatter_tensor, all_reduce
from lightllm.common.basemodel.triton_kernel.sp_pad_copy import sp_pad_copy


class BaseLayerInfer:
    def __init__(self) -> None:
        self.tp_rank_ = get_current_rank_in_dp()
        self.tp_world_size_ = get_dp_world_size()

    def context_forward(self, input: torch.Tensor, infer_state: InferStateInfo, layer_weight: BaseLayerWeight):
        raise Exception("need to impl")

    def token_forward(self, input: torch.Tensor, infer_state: InferStateInfo, layer_weight: BaseLayerWeight):
        raise Exception("need to impl")

    def alloc_tensor(
        self,
        shape: Union[torch.Size, Iterable[int]],
        dtype: torch.dtype,
        device: str = "cuda",
    ) -> torch.Tensor:
        """ """
        return g_cache_manager.alloc_tensor(shape, dtype, device=device)

    def overlap_tpsp_token_forward(
        self,
        input0: torch.Tensor,
        input1: torch.Tensor,
        infer_state: InferStateInfo,
        infer_state1: InferStateInfo,
        layer_weight: BaseLayerWeight,
    ):
        raise Exception("need to impl")

    def overlap_tpsp_context_forward(
        self,
        input0: torch.Tensor,
        input1: torch.Tensor,
        infer_state: InferStateInfo,
        infer_state1: InferStateInfo,
        layer_weight: BaseLayerWeight,
    ):
        raise Exception("need to impl")

    def _tpsp_allgather(self, input: torch.Tensor, infer_state: InferStateInfo):
        if self.tp_world_size_ > 1 and get_env_start_args().enable_tpsp_mix_mode:
            sp_token_num, hidden_dim = input.shape
            gather_input = self.alloc_tensor(
                (sp_token_num * self.tp_world_size_, hidden_dim), dtype=input.dtype, device=input.device
            )
            all_gather_into_tensor(gather_input, input, group=infer_state.dist_group, async_op=False)
            return gather_input
        return input

    def _tpsp_reduce(self, input: torch.Tensor, infer_state: InferStateInfo):
        """
        函数内部会根据当前的启动参数决定是进行reduce scatter还是all reduce
        """
        if self.tp_world_size_ > 1 and get_env_start_args().enable_tpsp_mix_mode:
            sp_token_num = input.shape[0] // self.tp_world_size_
            assert input.shape[0] % self.tp_world_size_ == 0
            hidden_dim = input.view(input.shape[0], -1).shape[1]
            reduce_o_tensor = self.alloc_tensor((sp_token_num, hidden_dim), dtype=input.dtype, device=input.device)
            reduce_scatter_tensor(
                output=reduce_o_tensor,
                input=input,
                op=dist.ReduceOp.SUM,
                group=infer_state.dist_group,
                async_op=False,
            )
            return reduce_o_tensor
        elif self.tp_world_size_ > 1:
            all_reduce(input, op=dist.ReduceOp.SUM, group=infer_state.dist_group, async_op=False)
            return input
        return input

    def _tpsp_sp_split(self, input: torch.Tensor, infer_state: InferStateInfo):
        """
        根据当前的启动参数决定是否将请求进行sp分割，如果需要分割，则进行分割，并返回分割后的结果
        如果不需要分割，则返回原始请求, 举列说明，如果input shape 为【16, 1024】，tp_world_size为4，则分割后返回的shape为【4, 1024】
        """
        if self.tp_world_size_ > 1 and get_env_start_args().enable_tpsp_mix_mode:
            input = sp_pad_copy(
                in_tensor=input,
                sp_rank_id=self.tp_rank_,
                sp_world_size=self.tp_world_size_,
                alloc_func=self.alloc_tensor,
            )
            return input
        return input
