import uuid
import numpy as np
from ...batch import Batch, Req
from lightllm.server.router.req_queue.base_queue import BaseQueue
from lightllm.common.basemodel.infer_lock import g_router_lock


class ChunkedPrefillQueue(BaseQueue):
    def __init__(self, args, router, dp_index, dp_size_in_node) -> None:
        super().__init__(args, router, dp_index, dp_size_in_node)
        self.batch_max_tokens = self.batch_max_tokens * 2

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

    # @calculate_time(show=True, min_cost_ms=0.1)
    def _can_add_new_req(self, req: Req, is_busy, new_batch_first_router_need_tokens):
        self.cache_len_list.append(req.get_tuple_tokens(is_busy, self.router_max_new_token_len))  # hard to analysis
        self.cache_len_list.sort(key=lambda x: -x[1])

        left_out_len_array = np.array([e[1] for e in self.cache_len_list])
        has_run_len_array = np.array([e[0] for e in self.cache_len_list])
        cum_run_len_array = np.cumsum(has_run_len_array)
        size_array = np.arange(1, len(self.cache_len_list) + 1, 1)

        need_max_token_num = (left_out_len_array * size_array + cum_run_len_array).max()
        with g_router_lock.obj:
            ok_token_num = (
                need_max_token_num + self.router.shared_token_load.get_frozened_token_count(self.dp_index)
                < self.max_total_tokens
            )

            ok_req_num = len(self.cache_len_list) <= self.running_max_req_size

            new_batch_first_router_need_tokens += req.get_first_router_need_tokens()
            ok_prefill = new_batch_first_router_need_tokens <= self.batch_max_tokens

            if ok_token_num and ok_req_num and ok_prefill:
                self.router.shared_token_load.set_estimated_peak_token_count(need_max_token_num, self.dp_index)
                self.router.shared_token_load.set_dynamic_max_load(
                    (need_max_token_num + self.router.shared_token_load.get_frozened_token_count(self.dp_index))
                    / self.max_total_tokens,
                    self.dp_index,
                )
                return True, new_batch_first_router_need_tokens
            else:
                return False, new_batch_first_router_need_tokens

    # @calculate_time(show=True, min_cost_ms=10)
    def generate_new_batch(self, current_batch: Batch):
        if len(self.waiting_req_list) == 0:
            return None

        # 如果当前已经被调度的请求数量超过了上限，直接不调度新的请求了。
        exist_req_num = self.get_batch_dp_req_size(current_batch)
        req_is_full = exist_req_num >= self.running_max_req_size
        if req_is_full:
            return None

        is_busy = self.is_busy()

        new_batch_first_router_need_tokens = (
            0 if current_batch is None else current_batch.get_batch_decode_need_tokens()[self.dp_index]
        )

        self._init_cache_list(current_batch, is_busy)
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
            ok_insert, new_batch_first_router_need_tokens = self._can_add_new_req(
                req, is_busy, new_batch_first_router_need_tokens
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
        is_busy = self.is_busy()
        self._init_cache_list(current_batch, is_busy)
        if len(self.cache_len_list) != 0:
            self.cache_len_list.sort(key=lambda x: -x[1])
            left_out_len_array = np.array([e[1] for e in self.cache_len_list])
            has_run_len_array = np.array([e[0] for e in self.cache_len_list])
            cum_run_len_array = np.cumsum(has_run_len_array)
            size_array = np.arange(1, len(self.cache_len_list) + 1, 1)
            need_max_token_num = (left_out_len_array * size_array + cum_run_len_array).max()
        else:
            need_max_token_num = 0

        with g_router_lock.obj:
            return (
                need_max_token_num,
                (need_max_token_num + self.router.shared_token_load.get_frozened_token_count(self.dp_index))
                / self.max_total_tokens,
            )
