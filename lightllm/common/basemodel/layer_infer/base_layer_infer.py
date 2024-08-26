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

    # 后续的 mark_cache_alloc_start, mark_cache_alloc_end, alloc_tensor， release_all_caches
    # 4 个函数接口，理论上只能 transformer 层进行调用。不要在其他层进行调用, 用户只需要调用 alloc_tensor 即可
    # 其他接口是在 basemodel.py 的框架流程中进行调用的
    def mark_cache_alloc_start(self):
        g_cache_manager.mark_cache_alloc_start()

    def mark_cache_alloc_end(self):
        g_cache_manager.mark_cache_alloc_end()

    def alloc_tensor(
        self, shape: Union[torch.Size, Iterable[int]], data_type: torch.dtype, device: str = "cuda"
    ) -> torch.Tensor:
        return g_cache_manager.alloc_tensor(shape, data_type, device=device)

    def release_all_caches(self):
        g_cache_manager.release_all_caches()
        return
