import torch
from typing import Dict, Iterable, Literal, Tuple, Union, List
from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.common.basemodel.layer_weights.base_layer_weight import BaseLayerWeight
from lightllm.utils.dist_utils import get_current_rank_in_dp, get_dp_world_size
from .cache_tensor_manager import g_cache_manager


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
        is_graph_out: bool = False,
        microbatch_index: int = 0,
    ) -> torch.Tensor:
        """
        is_graph_out 用于标记是graph图推理中的最后一个tensor，该参数只会在开启cuda graph时生效。
        该tensor的复用有特殊的逻辑，用于降低显存占用。
        microbatch_index 参数是为了支持microbatch overlap 模式所添加的参数，其值只能为0或者1，用以
        标记申请的tensor是用于第几个microbatch的，当前这个参数只有在 is_graph_out 为 True的时候会有
        具体的意义，其他情况没有实际意义。
        """
        return g_cache_manager.alloc_tensor(
            shape, dtype, device=device, is_graph_out=is_graph_out, microbatch_index=microbatch_index
        )

    def tpsp_context_forward(self, input: torch.Tensor, infer_state: InferStateInfo, layer_weight: BaseLayerWeight):
        raise Exception("need to impl")

    def tpsp_token_forward(self, input: torch.Tensor, infer_state: InferStateInfo, layer_weight: BaseLayerWeight):
        raise Exception("need to impl")

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
