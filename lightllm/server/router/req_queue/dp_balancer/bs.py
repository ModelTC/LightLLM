import os
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

    def __init__(
        self,
        dp_size_in_node: int,
        inner_queues: List[BaseQueue],
        balance_by_input_tokens: bool = False,
    ):
        super().__init__(dp_size_in_node, inner_queues)
        self.balance_by_input_tokens = balance_by_input_tokens
        self._stats_every = max(0, int(os.getenv("LIGHTLLM_DP_BALANCE_LOG_EVERY", "0")))
        self._assigned_group_count = 0
        self._assigned_req_count = [0 for _ in range(dp_size_in_node)]
        self._assigned_load = [0 for _ in range(dp_size_in_node)]

    def _req_load(self, req) -> int:
        if not self.balance_by_input_tokens:
            return 1
        return max(1, req.input_len - max(0, req.shm_cur_kv_len))

    def _queue_load(self, reqs: List) -> int:
        return sum(self._req_load(req) for req in reqs)

    def assign_reqs_to_dp(self, current_batch: Batch, reqs_waiting_for_dp_index: List[List[Req]]) -> None:
        if len(reqs_waiting_for_dp_index) == 0:
            return
        # calculate the total load of each dp rank
        current_load_per_dp = [0 for _ in range(self.dp_size_in_node)]
        if current_batch is not None:
            current_load_per_dp = [
                self._queue_load(current_batch.get_req_list_for_dp(i)) for i in range(self.dp_size_in_node)
            ]
        total_load_per_dp = [
            current_load_per_dp[i] + self._queue_load(self.inner_queues[i].waiting_req_list)
            for i in range(self.dp_size_in_node)
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
            group_load = self._queue_load(req_group)
            total_load_per_dp[suggested_dp_index] += group_load

            if self._stats_every:
                self._assigned_group_count += 1
                self._assigned_req_count[suggested_dp_index] += len(req_group)
                self._assigned_load[suggested_dp_index] += group_load
                if self._assigned_group_count % self._stats_every == 0:
                    logger.info(
                        "dp_balance_stats mode=%s groups=%d requests=%s load=%s queued_load=%s",
                        "input_tokens" if self.balance_by_input_tokens else "requests",
                        self._assigned_group_count,
                        self._assigned_req_count,
                        self._assigned_load,
                        total_load_per_dp,
                    )

        reqs_waiting_for_dp_index.clear()
        return
