import uuid
import numpy as np
from typing import List, Tuple
from .batch import Batch, Req
from lightllm.server.router.req_queue.base_queue import BaseQueue
from lightllm.server.router.req_queue.dp_base_queue import DpQueue


class Strategy:
    def ordering_reqs(self, batch: Batch):
        raise not NotImplemented()


class Fcfs(Strategy):
    def __init__(self) -> None:
        super().__init__()

    def ordering_reqs(self, reqs: List):
        return reqs[::-1]


def select_paused_reqs(
    batch: Batch, strategy: Strategy, req_queue: BaseQueue, max_total_token_num: int, dp_index: int
) -> List[Req]:
    if isinstance(req_queue, DpQueue):
        req_queue = req_queue.get_dp_queue(dp_index)
    reqs: List[Req] = strategy.ordering_reqs(batch.get_req_list_for_dp(dp_index))

    if len(reqs) == 0:
        return []

    group_req_id = reqs[0].group_req_id
    pause_reqs = []
    for req in reqs:
        if req.group_req_id == group_req_id:
            pause_reqs.append(req)
            batch.pop_req(req.request_id)
        else:
            break

    # 更新请求状态
    for req in pause_reqs:
        req.is_paused = True

    req_queue.back_to_wait_list(pause_reqs)

    return pause_reqs
