import time
import uuid
import numpy as np
from typing import List
from lightllm.utils.infer_utils import calculate_time
from lightllm.server.io_struct import Batch, Req
from lightllm.server.io_struct import ReqRunStatus
from lightllm.server.router.req_queue.base_queue import BaseQueue


class BeamContinuesBatchQueue(BaseQueue):
    def __init__(self, args, router) -> None:
        super().__init__(args, router)
        assert args.use_dynamic_prompt_cache is False
        return

    def _init_cache_list(self, current_batch: Batch, is_busy):
        if current_batch is not None:
            self.cache_len_list = [
                (req, req.get_tuple_tokens(is_busy, self.router_max_new_token_len)) for req in current_batch.reqs
            ]
        else:
            self.cache_len_list = []
        return

    # @calculate_time(show=True, min_cost_ms=0.1)
    def _can_add_new_group_reqs(self, cur_handle_group_reqs: List[Req], is_busy, new_batch_first_router_need_tokens):
        for req in cur_handle_group_reqs:
            self.cache_len_list.append(
                (req, req.get_tuple_tokens(is_busy, self.router_max_new_token_len))
            )  # hard to analysis

        self.cache_len_list.sort(key=lambda x: -x[1][1])

        need_max_token_num = 0
        cumsum_len = 0
        exist_group_req_set = set()
        for index, (req, (cur_input_len, cur_ouput_len)) in enumerate(self.cache_len_list, 1):
            if req.group_req_id not in exist_group_req_set:
                exist_group_req_set.add(req.group_req_id)
                cumsum_len += cur_input_len
                need_max_token_num = max(need_max_token_num, cumsum_len + index * cur_ouput_len)
            else:
                # 因为有共享的token，所以
                assert cur_input_len - req.input_len >= 0
                cumsum_len += cur_input_len - req.input_len  # 减去共享的部分
                need_max_token_num = max(need_max_token_num, cumsum_len + index * cur_ouput_len)

        # prefill token 计算
        for req in cur_handle_group_reqs:
            new_batch_first_router_need_tokens += req.cur_output_len
        new_batch_first_router_need_tokens += req.input_len

        ok_token_num = need_max_token_num < self.max_total_tokens

        if req.req_status != ReqRunStatus.PAUSED_AND_OFFLOAD:
            ok_req_num = len(self.cache_len_list) + len(self.pause_req_dict) <= self.running_max_req_size
        else:
            ok_req_num = (
                len(self.cache_len_list) + len(self.pause_req_dict) - len(cur_handle_group_reqs)
                <= self.running_max_req_size
            )

        # prefill ok
        ok_prefill = new_batch_first_router_need_tokens <= self.batch_max_tokens

        if ok_token_num and ok_req_num and ok_prefill:
            self.router.shared_token_load.set_dynamic_max_load(need_max_token_num / self.max_total_tokens)
            return True, new_batch_first_router_need_tokens
        else:
            return False, new_batch_first_router_need_tokens

    # @calculate_time(show=True, min_cost_ms=10)
    def generate_new_batch(self, current_batch: Batch):
        # 如果当前已经被调度的请求数量超过了上限，直接不调度新的请求了。
        exist_req_num = (0 if current_batch is None else len(current_batch.reqs)) + len(self.pause_req_dict)
        req_is_full = exist_req_num >= self.running_max_req_size
        if req_is_full:
            return None
        if len(self.waiting_req_list) == 0:
            return None

        # 判断服务是否繁忙
        is_busy = self.is_busy()

        self._init_cache_list(current_batch, is_busy)
        can_run_list = []
        new_batch_first_router_need_tokens = 0  # 主要是对 prefill 大块计算时候的token数量限制
        aborted_count = 0
        cur_group_reqs = []
        for req in self.waiting_req_list:
            if req.finish_status.is_aborted() and req.req_status == ReqRunStatus.WAIT_IN_QUEUE:
                aborted_count += 1
                continue

            if self._add_to_group(cur_group_reqs, req):
                continue

            ok_insert, new_batch_first_router_need_tokens = self._can_add_new_group_reqs(
                cur_group_reqs, is_busy, new_batch_first_router_need_tokens
            )
            if ok_insert:
                can_run_list.extend(cur_group_reqs)
                for cur_req in cur_group_reqs:
                    if cur_req.req_status == ReqRunStatus.PAUSED_AND_OFFLOAD:
                        self.pause_req_dict.pop(cur_req.request_id)
                cur_group_reqs = [req]  # 等待判断的组
            else:
                cur_group_reqs = []
                break

        if len(cur_group_reqs) != 0:
            ok_insert, new_batch_first_router_need_tokens = self._can_add_new_group_reqs(
                cur_group_reqs, is_busy, new_batch_first_router_need_tokens
            )
            if ok_insert:
                can_run_list.extend(cur_group_reqs)
                for req in cur_group_reqs:
                    if req.req_status == ReqRunStatus.PAUSED_AND_OFFLOAD:
                        self.pause_req_dict.pop(req.request_id)

        if len(can_run_list) != 0:
            new_batch = Batch(uuid.uuid4().hex, can_run_list)
            self.waiting_req_list = self.waiting_req_list[len(can_run_list) + aborted_count :]
            return new_batch
        else:
            return None

    def _add_to_group(self, cur_group_reqs, req: Req):
        if len(cur_group_reqs) == 0:
            cur_group_reqs.append(req)
            return True
        else:
            if req.group_req_id == cur_group_reqs[-1].group_req_id:
                cur_group_reqs.append(req)
                return True
            else:
                return False

    def calcu_batch_token_load(self, current_batch: Batch):
        if current_batch is None:
            return 0.0
        is_busy = self.is_busy()
        self._init_cache_list(current_batch, is_busy)
        self.cache_len_list.sort(key=lambda x: -x[1][1])
        need_max_token_num = 0
        cumsum_len = 0
        exist_group_req_set = set()
        for index, (req, (cur_input_len, cur_ouput_len)) in enumerate(self.cache_len_list, 1):
            if req.group_req_id not in exist_group_req_set:
                exist_group_req_set.add(req.group_req_id)
                cumsum_len += cur_input_len
                need_max_token_num = max(need_max_token_num, cumsum_len + index * cur_ouput_len)
            else:
                # 因为有共享的token
                assert cur_input_len - req.input_len >= 0
                cumsum_len += cur_input_len - req.input_len  # 减去共享的部分
                need_max_token_num = max(need_max_token_num, cumsum_len + index * cur_ouput_len)
        return need_max_token_num / self.max_total_tokens
