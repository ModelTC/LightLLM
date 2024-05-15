import uuid
import asyncio
import numpy as np
from typing import List
from lightllm.utils.infer_utils import calculate_time
from lightllm.server.io_struct import Batch, Req
from lightllm.server.io_struct import ReqRunStatus, FinishStatus


class BaseQueue:
    def __init__(self, args, router) -> None:
        self.args = args
        from lightllm.server.router.manager import RouterManager

        self.router: RouterManager = router
        self.max_total_tokens = args.max_total_token_num
        assert args.batch_max_tokens is not None
        self.batch_max_tokens = args.batch_max_tokens
        self.running_max_req_size = args.running_max_req_size  # 最大并非请求数量
        self.waiting_req_list: List[Req] = []  # 当前等待队列
        self.router_token_ratio = args.router_token_ratio  # 调度繁忙
        self.router_max_new_token_len = args.router_max_new_token_len
        self.pause_req_dict = {}  # 用于保存队列中被暂停的请求，暂停原因为 ReqRunStatus.PAUSED_AND_OFFLOAD

    def append(self, req):
        self.waiting_req_list.append(req)
        return

    def extend(self, req_group: List):
        self.waiting_req_list.extend(req_group)
        return

    def back_to_wait_list(self, req_list: List[Req]):
        for req in req_list:
            if req.req_status in [
                ReqRunStatus.PAUSED_AND_OFFLOAD,
            ]:
                self.pause_req_dict[req.request_id] = req
        self.waiting_req_list = req_list + self.waiting_req_list
        return

    def is_busy(self):
        # 计算当前所有的token使用量, 如果使用了dynamic prompt cache, 使用的token量中不包含，cache tree 中未被引用的数据。
        cur_all_used_tokens = self.router.get_used_tokens()
        # 判断当前服务是否处于token使用率过高的状态，过高的情况下，调度要偏向保守
        cur_token_ratio = cur_all_used_tokens / self.max_total_tokens
        is_busy = cur_token_ratio >= self.router_token_ratio
        return is_busy

    def generate_new_batch(self, current_batch: Batch):
        raise NotImplementedError()

    def calcu_batch_token_load(self, current_batch: Batch):
        raise NotImplementedError()

    def update_token_load(self, current_batch: Batch):
        if self.router.shared_token_load.need_update_dynamic_max_load():
            self.router.shared_token_load.set_dynamic_max_load(self.calcu_batch_token_load(current_batch))
        return
