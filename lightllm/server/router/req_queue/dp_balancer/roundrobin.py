import random
from typing import List, Union
from lightllm.server.router.req_queue.base_queue import BaseQueue
from lightllm.server.router.batch import Batch, Req
from lightllm.utils.log_utils import init_logger
from .base import DpBalancer

logger = init_logger(__name__)


class RoundRobinDpBalancer(DpBalancer):
    """
    轮询负载均衡器
    在队列长度最小的DP中进行轮询选择
    """

    def __init__(self, dp_size_in_node: int, inner_queues: List[BaseQueue]):
        super().__init__(dp_size_in_node, inner_queues)
        self.pre_select_dp_index = self.dp_size_in_node - 1

    def get_suggest_dp_index(
        self,
    ) -> int:
        min_length = min(len(queue.waiting_req_list) for queue in self.inner_queues)
        select_dp_indexes = [
            i for i, queue in enumerate(self.inner_queues) if len(queue.waiting_req_list) == min_length
        ]

        # 如果没有可选择的索引，随机选择一个
        if not select_dp_indexes:
            self.pre_select_dp_index = random.randint(0, self.dp_size_in_node - 1)
            return self.pre_select_dp_index

        # 轮询选择
        for i in range(self.dp_size_in_node):
            next_dp_index = (self.pre_select_dp_index + i + 1) % self.dp_size_in_node
            if next_dp_index in select_dp_indexes:
                self.pre_select_dp_index = next_dp_index
                return self.pre_select_dp_index

        self.pre_select_dp_index = random.choice(select_dp_indexes)
        return self.pre_select_dp_index

    def assign_reqs_to_dp(self, current_batch: Batch, reqs_waiting_for_dp_index: List[List[Req]]) -> None:
        for req_group in reqs_waiting_for_dp_index:
            suggested_dp_index = self.get_suggest_dp_index()
            for req in req_group:
                req.sample_params.suggested_dp_index = suggested_dp_index
            self.inner_queues[suggested_dp_index].extend(req_group)
        reqs_waiting_for_dp_index.clear()
        return
