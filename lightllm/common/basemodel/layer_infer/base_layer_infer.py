import torch
from typing import Dict, Iterable, Literal, Tuple, Union, List
from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.common.basemodel.splitfuse_infer_struct import SplitFuseInferStateInfo
from lightllm.common.basemodel.layer_weights.base_layer_weight import BaseLayerWeight
from .cache_tensor_manager import g_cache_manager


class BaseLayerInfer:
    def __init__(self) -> None:
        pass

    def context_forward(self, input_ids, infer_state: InferStateInfo, layer_weight: BaseLayerWeight):
        raise Exception("need to impl")

    def token_forward(self, input_ids, infer_state: InferStateInfo, layer_weight: BaseLayerWeight):
        raise Exception("need to impl")

    def splitfuse_forward(self, input_ids, infer_state: SplitFuseInferStateInfo, layer_weight: BaseLayerWeight):
        raise Exception("need to impl")

    def alloc_tensor(
        self,
        shape: Union[torch.Size, Iterable[int]],
        data_type: torch.dtype,
        device: str = "cuda",
        is_graph_out: bool = False,
    ) -> torch.Tensor:
        """
        is_graph_out 用于标记是graph图推理中的最后一个tensor，该参数只会在开启cuda graph时生效。该tensor的复用有特殊的逻辑，用于降低显存
        占用
        """
        return g_cache_manager.alloc_tensor(shape, data_type, device=device, is_graph_out=is_graph_out)
