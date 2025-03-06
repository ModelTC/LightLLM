import time
import uuid
import numpy as np
from typing import List
from lightllm.utils.infer_utils import calculate_time
from ...batch import Batch, Req
from lightllm.server.router.req_queue.base_queue import BaseQueue
from lightllm.common.basemodel.infer_lock import g_router_lock


class ContinuesBatchQueueForPDDecode(BaseQueue):
    def __init__(self, args, router, dp_index, dp_size_in_node) -> None:
        super().__init__(args, router, dp_index, dp_size_in_node)

    def _init_cache_list(self, current_batch: Batch, is_busy):
        if current_batch is not None:
            self.cache_len_list = [
                req.get_tuple_tokens(is_busy, self.router_max_new_token_len)
                for req in current_batch.reqs
                if req.sample_params.suggested_dp_index == self.dp_index
            ]
        else:
            self.cache_len_list = []
        return

    # @calculate_time(show=True, min_cost_ms=10)
    def generate_new_batch(self, current_batch: Batch, limit_router_queue_length: int = None):
        # 如果当前已经被调度的请求数量超过了上限，直接不调度新的请求了。
        exist_req_num = self.get_batch_dp_req_size(current_batch) + len(self.pause_req_dict)
        req_is_full = exist_req_num >= self.running_max_req_size
        if req_is_full:
            return None

        can_run_list = []
        abort_req_list = []
        aborted_count = 0
        for req in self.waiting_req_list:
            if req.is_aborted and not req.is_paused:
                # 由于管理的复杂性，只有没有被调度运行过的请求可以因为abort直接在队列中忽略掉.
                # 暂停的请求需要恢复后，由 router manager 部分来过滤。暂时保持这种处理方法, 否则会导致管理token和管理req对象的泄漏
                aborted_count += 1
                abort_req_list.append(req)
                continue
            if exist_req_num + len(can_run_list) + 1 <= self.batch_max_tokens:
                can_run_list.append(req)
            else:
                break

        if len(can_run_list) != 0:
            new_batch = Batch(uuid.uuid4().int, can_run_list, dp_size_in_node=self.dp_size_in_node)
            for req in abort_req_list:
                self.router.shm_req_manager.put_back_req_obj(req)
            self.waiting_req_list = self.waiting_req_list[len(can_run_list) + aborted_count :]
            return new_batch
        else:
            return None

    def _calcu_batch_token_load_batch_not_none(self, current_batch: Batch):
        is_busy = self.is_busy()
        self._init_cache_list(current_batch, is_busy)
        self.cache_len_list.sort(key=lambda x: -x[1])
        left_out_len_array = np.array([e[1] for e in self.cache_len_list])
        has_run_len_array = np.array([e[0] for e in self.cache_len_list])
        cum_run_len_array = np.cumsum(has_run_len_array)
        size_array = np.arange(1, len(self.cache_len_list) + 1, 1)
        need_max_token_num = (left_out_len_array * size_array + cum_run_len_array).max()
        return (
            need_max_token_num,
            (need_max_token_num + self.router.shared_token_load.get_frozened_token_count(self.dp_index))
            / self.max_total_tokens,
        )
