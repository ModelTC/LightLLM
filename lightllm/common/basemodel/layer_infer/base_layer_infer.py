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
    ) -> torch.Tensor:
        """ """
        return g_cache_manager.alloc_tensor(shape, dtype, device=device)

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
