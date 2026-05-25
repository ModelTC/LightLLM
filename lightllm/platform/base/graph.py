from abc import ABC, abstractmethod
from typing import Any, ContextManager, Optional


class BackendGraph(ABC):

    @abstractmethod
    def create_graph(self) -> Any:
        pass

    @abstractmethod
    def graph(self, graph_obj: Any, pool: Optional[Any] = None, stream: Optional[Any] = None) -> ContextManager:
        pass

    def replay_graph(self, graph_obj: Any) -> Any:
        graph_obj.replay()

    @abstractmethod
    def graph_pool_handle(self) -> Any:
        pass

    @abstractmethod
    def is_capturing(self) -> bool:
        pass
