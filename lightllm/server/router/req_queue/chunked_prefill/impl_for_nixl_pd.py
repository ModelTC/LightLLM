import uuid
import numpy as np
from typing import Tuple
from ...batch import Batch, Req
from lightllm.server.router.req_queue.base_queue import BaseQueue
from lightllm.common.basemodel.infer_lock import g_router_lock


class NIXLPDQueue(BaseQueue):
    def __init__(self, args, router, dp_index, dp_size_in_node) -> None:
        super().__init__(args, router, dp_index, dp_size_in_node)

    # @calculate_time(show=True, min_cost_ms=0.1)
    def _can_add_new_req(self, req: Req, estimated_peak_token_num: int, batch_req_num: int) -> Tuple[bool, int, int]:
        estimated_peak_token_num += req.input_len + req.sample_params.max_new_tokens
        ok_token_num = estimated_peak_token_num < self.max_total_tokens
        batch_req_num += 1
        ok_req_num = batch_req_num <= self.running_max_req_size

        if ok_token_num and ok_req_num:
            self.router.shared_token_load.set_estimated_peak_token_count(estimated_peak_token_num, self.dp_index)
            self.router.shared_token_load.set_dynamic_max_load(
                estimated_peak_token_num / self.max_total_tokens,
                self.dp_index,
            )
            return True, estimated_peak_token_num, batch_req_num
        else:
            return False, None, None

    def _caclu_batch_estimated_peak_token_num(self, batch: Batch):
        is_busy = self.is_busy()
        estimated_peak_token_num = 0
        decoding_req_list = []
        if batch is not None:
            for req in batch.reqs:
                if req.sample_params.suggested_dp_index == self.dp_index:
                    if req.is_infer_decode():
                        decoding_req_list.append(req.get_tuple_tokens(is_busy, self.router_max_new_token_len))
                    else:
                        estimated_peak_token_num += req.input_len + req.sample_params.max_new_tokens

        if decoding_req_list:
            decoding_req_list.sort(key=lambda x: -x[1])
            left_out_len_array = np.array([e[1] for e in decoding_req_list])
            has_run_len_array = np.array([e[0] for e in decoding_req_list])
            cum_run_len_array = np.cumsum(has_run_len_array)
            size_array = np.arange(1, len(decoding_req_list) + 1, 1)
            estimated_peak_token_num += (left_out_len_array * size_array + cum_run_len_array).max()

        return estimated_peak_token_num

    # @calculate_time(show=True, min_cost_ms=10)
    def generate_new_batch(self, current_batch: Batch):
        if len(self.waiting_req_list) == 0:
            return None

        # 如果当前已经被调度的请求数量超过了上限，直接不调度新的请求了。
        exist_req_num = self.get_batch_dp_req_size(current_batch)
        req_is_full = exist_req_num >= self.running_max_req_size
        if req_is_full:
            return None

        estimated_peak_token_num = self._caclu_batch_estimated_peak_token_num(current_batch)
        batch_req_num = exist_req_num

        can_run_list = []
        abort_req_list = []
        aborted_count = 0

        waiting_queue = self.waiting_req_list

        for req in waiting_queue:
            if req.is_aborted:
                # 由于管理的复杂性，只有没有被调度运行过的请求可以因为abort直接在队列中忽略掉.
                # 暂停的请求需要恢复后，由 router manager 部分来过滤。暂时保持这种处理方法, 否则会导致管理token的泄漏
                aborted_count += 1
                abort_req_list.append(req)
                continue
            ok_insert, estimated_peak_token_num, batch_req_num = self._can_add_new_req(
                req=req, estimated_peak_token_num=estimated_peak_token_num, batch_req_num=batch_req_num
            )
            if ok_insert:
                can_run_list.append(req)
            else:
                break
        new_batch = None
        if len(can_run_list) != 0:
            new_batch = Batch(uuid.uuid4().int, can_run_list, dp_size_in_node=self.dp_size_in_node)
        for req in abort_req_list:
            self.router.shm_req_manager.put_back_req_obj(req)
        self.waiting_req_list = self.waiting_req_list[len(can_run_list) + aborted_count :]
        return new_batch

    def _calcu_batch_token_load_batch_not_none(self, current_batch: Batch):

        estimated_peak_token_num = self._caclu_batch_estimated_peak_token_num(current_batch)

        return (estimated_peak_token_num, estimated_peak_token_num / self.max_total_tokens)
