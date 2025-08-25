import random
from abc import ABC, abstractmethod
from typing import List, Union
from lightllm.server.router.req_queue.base_queue import BaseQueue
from lightllm.server.router.batch import Batch, Req
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class DpBalancer(ABC):
    """
    DP负载均衡器基类
    定义了负载均衡策略的接口，子类可以实现不同的负载均衡算法
    """

    def __init__(self, dp_size_in_node: int, inner_queues: List[BaseQueue]):
        self.dp_size_in_node = dp_size_in_node
        self.inner_queues = inner_queues

    @abstractmethod
    def assign_reqs_to_dp(self, current_batch: Batch, reqs_waiting_for_dp_index: List[List[Req]]) -> None:
        pass
