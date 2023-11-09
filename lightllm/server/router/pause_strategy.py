import uuid
import numpy as np
from typing import List, Tuple
from lightllm.server.io_struct import Batch, Req
from lightllm.server.router.req_queue import ReqQueue
from lightllm.server.io_struct import ReqRunStatus

class Strategy:

    def ordering_reqs(self, batch: Batch):
        raise not NotImplemented()

class Fcfs(Strategy):

    def __init__(self) -> None:
        super().__init__()

    def ordering_reqs(self, batch: Batch):
        reqs = [req for req in batch.reqs]
        return sorted(reqs, key=lambda req: req.request_id, reverse=True)

class Sfj(Strategy):

    def __init__(self) -> None:
        super().__init__()

    def ordering_reqs(self, batch: Batch):
        reqs = [req for req in batch.reqs]
        return sorted(reqs, key=lambda req: req.max_output_len - len(req.output_ids), reverse=True)

class Hrnn(Strategy):

    def __init__(self) -> None:
        super().__init__()
    
    def ordering_reqs(self, batch: Batch):
        reqs = [req for req in batch.reqs]
        return sorted(reqs, key=lambda req: (req.input_len + req.max_output_len - len(req.output_ids)) / req.input_len, reverse=True)


def select_paused_reqs(batch: Batch, strategy: Strategy, req_queue: ReqQueue, max_total_token_num):
    reqs = strategy.ordering_reqs(batch)
    pause_req : Req = reqs[0]
    batch.pop_req(pause_req.request_id)

    pause_req.req_status = ReqRunStatus.PAUSED_AND_OFFLOAD
    pause_req.offload_kv_len = pause_req.input_len + len(pause_req.output_ids) - 1
    req_queue.back_to_wait_list([pause_req])

    return [pause_req]

