import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from lightllm.server.core.objs import ShmReqManager, Req
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class Batch:
    def __init__(self, batch_id, reqs: List[Req], dp_size: int):
        self.batch_id = batch_id
        self.reqs = reqs
        self.id_to_reqs = {req.request_id: req for req in reqs}
        self.dp_size = dp_size
        return

    def input_tokens(self):
        batch_input_tokens = 0
        for req in self.reqs:
            batch_input_tokens += req.input_len
        return batch_input_tokens

    def get_batch_decode_need_tokens(self):
        new_batch_decode_need_tokens = [0 for _ in range(self.dp_size)]  # 只有在 splitfuse 模式下有意义

        for req in self.reqs:
            req_dp_index = req.sample_params.suggested_dp_index
            new_batch_decode_need_tokens[req_dp_index] += req.get_decode_need_tokens()

        return new_batch_decode_need_tokens

    def filter_out_finished_req(self, shm_req_manager: ShmReqManager):
        unfinished_req_ids = []
        for req in self.reqs:
            # 更新aborted 标记，可以触发推理进程主动退出aborted得请求。
            if req.is_aborted:
                req.router_aborted = True

            if req.shm_infer_released:
                logger.info(f"router release req id {req.request_id}")
                shm_req_manager.put_back_req_obj(req)
                req = None
            else:
                unfinished_req_ids.append(req.request_id)

        self.reqs = [self.id_to_reqs[req_id] for req_id in unfinished_req_ids]
        self.id_to_reqs = {req.request_id: req for req in self.reqs}
        return

    def pop_req(self, req_id):
        self.reqs = [req for req in self.reqs if req.request_id != req_id]
        self.id_to_reqs.pop(req_id)
        return

    def is_clear(self):
        return len(self.reqs) == 0

    def merge(self, mini_batch: "Batch"):
        for _req in mini_batch.reqs:
            self.reqs.append(_req)
        self.id_to_reqs = {req.request_id: req for req in self.reqs}
        return

    def dp_merge(self, mini_batch: "Batch"):
        if mini_batch is None:
            return

        for _req in mini_batch.reqs:
            self.reqs.append(_req)
        self.id_to_reqs = {req.request_id: req for req in self.reqs}
        return

    def __repr__(self):
        return f"batch_id={self.batch_id}, " f"reqs={self.reqs}, "

    def simple_log(self):
        return f"batch_id={self.batch_id}, time:{time.time()}s req_ids:{[req.request_id for req in self.reqs]}"
