import torch
from typing import Any, ContextManager, Optional
from lightllm.platform.base.graph import BackendGraph


class AscendGraphBackend(BackendGraph):

    def create_graph(self) -> Any:
        return torch.npu.CUDAGraph()

    def graph(self, graph_obj: Any, pool: Optional[Any] = None, stream: Optional[Any] = None) -> ContextManager:
        return torch.npu.graph(graph_obj, pool=pool, stream=stream)

    def graph_pool_handle(self) -> Any:
        return torch.npu.graph_pool_handle()

    def is_capturing(self) -> bool:
        return torch.npu.is_current_stream_capturing()