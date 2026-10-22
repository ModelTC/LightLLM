import uuid
import numpy as np
from ...batch import Batch, Req
from lightllm.server.router.req_queue.base_queue import BaseQueue


class ChunkedPrefillQueue(BaseQueue):
    def _init_cache_list(self, current_batch: Batch, is_busy):
        if current_batch is not None:
            self.cache_len_list = [
                req.get_tuple_tokens(is_busy, self.router.router_statics.ema_req_out_len)
                for req in current_batch.reqs
                if req.sample_params.suggested_dp_index == self.dp_index
            ]
        else:
            self.cache_len_list = []
        return

    # @calculate_time(show=True, min_cost_ms=0.1)
    def _can_add_new_req(self, req: Req, is_busy):
        self.cache_len_list.append(
            req.get_tuple_tokens(is_busy, self.router.router_statics.ema_req_out_len)
        )  # hard to analysis
        self.cache_len_list.sort(key=lambda x: -x[1])

        left_out_len_array = np.array([e[1] for e in self.cache_len_list])
        has_run_len_array = np.array([e[0] for e in self.cache_len_list])
        cum_run_len_array = np.cumsum(has_run_len_array)
        size_array = np.arange(1, len(self.cache_len_list) + 1, 1)

        need_max_token_num = (left_out_len_array * size_array + cum_run_len_array).max()
        reserved = 0
        if getattr(self, "tool_call_admission_enabled", False):
            reservation_reader = getattr(self.router, "get_tool_call_reserved_tokens", None)
            if reservation_reader is not None:
                reserved = max(0, min(self.max_total_tokens, int(reservation_reader(self.dp_index))))
        # Idle tool prefixes are physically reclaimable, but admitting unrelated
        # work against all their space defeats the retention preference before
        # the continuation can reach inference. Account for that waiting space.
        capacity = self.max_total_tokens - reserved
        # Always allow a first feasible request to borrow the reservation. This
        # prevents head-of-line deadlock for a large request or a full budget.
        if (
            len(self.cache_len_list) == 1
            or getattr(req, "_tool_call_cache_hit", False)
            or getattr(getattr(req, "sample_params", None), "high_priority_request", False)
        ):
            capacity = self.max_total_tokens
        ok_token_num = need_max_token_num < capacity

        ok_req_num = len(self.cache_len_list) <= self.running_max_req_size

        if ok_token_num and ok_req_num:
            self.router.shared_token_load.set_estimated_peak_token_count(need_max_token_num, self.dp_index)
            self.router.shared_token_load.set_dynamic_max_load(
                need_max_token_num / self.max_total_tokens,
                self.dp_index,
            )
            return True
        else:
            return False

    def _refresh_tool_continuation_hints(self):
        reader = getattr(self.router, "get_tool_call_prefix_hints", None)
        hints = reader(self.dp_index) if reader is not None else None
        if hints is None:
            return
        limit = max(64, self.running_max_req_size * 2)
        for req in self.waiting_req_list[:limit]:
            req._tool_call_cache_hit = False
            if req.sample_params.high_priority_request:
                continue
            if hints.by_length and not getattr(req.sample_params, "disable_prompt_cache", False):
                if not hasattr(req, "shm_prompt_ids"):
                    req.link_prompt_ids_shm_array()
                memo = getattr(req, "_tool_call_prefix_memo", None)
                if memo is None:
                    req._tool_call_prefix_memo = memo = {}
                req._tool_call_cache_hit = hints.matches(req.get_prompt_ids_numpy(), memo)

    def _backfill_tool_continuations(self, waiting, blocked_index, is_busy, can_run_list):
        head = waiting[blocked_index]
        if getattr(head.sample_params, "high_priority_request", False):
            return
        if len(self.cache_len_list) >= self.running_max_req_size:
            return
        quota = max(0, 8 - getattr(head, "_tool_call_priority_skips", 0))
        limit = max(64, self.running_max_req_size * 2)
        for req in waiting[blocked_index + 1 : limit]:
            if quota == 0:
                break
            if not getattr(req, "_tool_call_cache_hit", False):
                continue
            before = self.cache_len_list[:]
            if self._can_add_new_req(req, is_busy):
                can_run_list.append(req)
                head._tool_call_priority_skips = getattr(head, "_tool_call_priority_skips", 0) + 1
                quota -= 1
            else:
                # Failed candidates must not inflate the next candidate's peak.
                self.cache_len_list = before

    # @calculate_time(show=True, min_cost_ms=10)
    def generate_new_batch(self, current_batch: Batch):
        self.filter_aborted_reqs()
        if not self.waiting_req_list:
            return None
        if self.get_batch_dp_req_size(current_batch) >= self.running_max_req_size:
            return None

        enabled = getattr(self, "tool_call_admission_enabled", False)
        if enabled:
            self._refresh_tool_continuation_hints()
        is_busy = self.is_busy()
        self._init_cache_list(current_batch, is_busy)
        can_run_list = []
        waiting = self.waiting_req_list
        for index, req in enumerate(waiting):
            before = self.cache_len_list[:] if enabled else None
            if self._can_add_new_req(req, is_busy):
                # A feasible FIFO head always runs before a later cached request.
                can_run_list.append(req)
            else:
                if enabled:
                    self.cache_len_list = before
                    # Only use space that the blocked head cannot currently use.
                    # At most eight successful cached admissions may bypass it.
                    self._backfill_tool_continuations(waiting, index, is_busy, can_run_list)
                break
        new_batch = None
        if can_run_list:
            new_batch = Batch(uuid.uuid4().int, can_run_list, dp_size_in_node=self.dp_size_in_node)
        if enabled:
            admitted = {id(req) for req in can_run_list}
            self.waiting_req_list = [req for req in waiting if id(req) not in admitted]
        else:
            self.waiting_req_list = waiting[len(can_run_list) :]
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

        return (need_max_token_num, need_max_token_num / self.max_total_tokens)
