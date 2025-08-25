import random
from typing import List, Union
from lightllm.server.router.req_queue.base_queue import BaseQueue
from lightllm.server.router.batch import Batch, Req
from lightllm.utils.log_utils import init_logger
from .base import DpBalancer

logger = init_logger(__name__)


class DpBsBalancer(DpBalancer):
    """
    This balancer is main to balance the batch size of each dp rank.
    Because, for dp mode, if it exists a dp rank without any request, it will
    padding a request and cause the waste of GPU compute resource.
    """

    def __init__(self, dp_size_in_node: int, inner_queues: List[BaseQueue]):
        super().__init__(dp_size_in_node, inner_queues)

    def assign_reqs_to_dp(self, current_batch: Batch, reqs_waiting_for_dp_index: List[List[Req]]) -> None:
        if len(reqs_waiting_for_dp_index) == 0:
            return
        # calculate the total load of each dp rank
        all_dp_req_num = [0 for _ in range(self.dp_size_in_node)]
        if current_batch is not None:
            all_dp_req_num = current_batch.get_all_dp_req_num()
        total_load_per_dp = [
            all_dp_req_num[i] + len(self.inner_queues[i].waiting_req_list) for i in range(self.dp_size_in_node)
        ]
        for req_group in reqs_waiting_for_dp_index:
            # find the dp rank with minimum load
            min_load = min(total_load_per_dp)
            select_dp_indexes = [i for i in range(self.dp_size_in_node) if total_load_per_dp[i] == min_load]
            suggested_dp_index = random.choice(select_dp_indexes)

            # assign the request to the dp rank and update the load count
            for req in req_group:
                req.sample_params.suggested_dp_index = suggested_dp_index
            self.inner_queues[suggested_dp_index].extend(req_group)
            # update the load count for this dp rank
            total_load_per_dp[suggested_dp_index] += len(req_group)

        reqs_waiting_for_dp_index.clear()
        return
