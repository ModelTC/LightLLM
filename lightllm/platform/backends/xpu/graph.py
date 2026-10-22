from typing import Any, ContextManager, Optional

import torch
from lightllm.platform.base.graph import HardwareBackendGraph


class XpuGraph(HardwareBackendGraph):
    def create_graph(self) -> Any:
        return torch.xpu.XPUGraph()

    def graph(
        self,
        graph_obj: Any,
        pool: Optional[Any] = None,
        stream: Optional[Any] = None,
    ) -> ContextManager:
        return torch.xpu.graph(graph_obj, pool=pool, stream=stream)

    def graph_pool_handle(self) -> Any:
        return torch.xpu.graph_pool_handle()

    def is_capturing(self) -> bool:
        return torch.xpu.is_current_stream_capturing()
