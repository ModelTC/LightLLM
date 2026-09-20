import uuid
from typing import Tuple
from ...batch import Batch, Req
from lightllm.server.router.req_queue.base_queue import BaseQueue
from lightllm.utils.log_utils import init_logger


logger = init_logger(__name__)


class PDPrefillQueue(BaseQueue):
    def __init__(self, args, router, dp_index, dp_size_in_node) -> None:
        super().__init__(args, router, dp_index, dp_size_in_node)
        logger.info(
            "PD prefill requests normally generate only one output token; "
            "estimate peak KV usage by summing each request's a_len + b_len, "
            "including page alignment, MTP and asynchronous exit reserves"
        )

    # @calculate_time(show=True, min_cost_ms=0.1)
    def _can_add_new_req(self, req: Req, estimated_peak_token_num: int, batch_req_num: int) -> Tuple[bool, int, int]:
        # 与已有 batch 使用相同的请求容量估算，确保新请求准入也计入分页、MTP 和异步退出所需的余量。
        a_len, b_len = req.get_tuple_tokens(self.is_busy(), self.router.router_statics.ema_req_out_len)
        estimated_peak_token_num = estimated_peak_token_num + (a_len + b_len)
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
        if batch is not None:
            for req in batch.reqs:
                if req.sample_params.suggested_dp_index == self.dp_index:
                    # PD prefill 请求通常只生成一个 token，其 KV 占用不会像 decode 请求一样持续增长，
                    # 因此直接累加每个请求的 a_len + b_len，作为整个 batch 的峰值容量估算。
                    # get_tuple_tokens 会结合当前输出长度和 KV 长度估算请求容量，并统一计入分页对齐、
                    # 两轮 MTP 以及 stop_str 等异步操作造成的退出延迟所需的余量，具体计算见该方法。
                    a_len, b_len = req.get_tuple_tokens(is_busy, self.router.router_statics.ema_req_out_len)
                    estimated_peak_token_num = estimated_peak_token_num + (a_len + b_len)

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

        self.filter_aborted_reqs()
        if len(self.waiting_req_list) == 0:
            return None

        estimated_peak_token_num = self._caclu_batch_estimated_peak_token_num(current_batch)
        batch_req_num = exist_req_num

        can_run_list = []
        consumed_req_count = 0

        waiting_queue = self.waiting_req_list

        for req in waiting_queue:
            ok_insert, estimated_peak_token_num, batch_req_num = self._can_add_new_req(
                req=req, estimated_peak_token_num=estimated_peak_token_num, batch_req_num=batch_req_num
            )
            if ok_insert:
                consumed_req_count += 1
                can_run_list.append(req)
            else:
                break
        new_batch = None
        if len(can_run_list) != 0:
            new_batch = Batch(uuid.uuid4().int, can_run_list, dp_size_in_node=self.dp_size_in_node)
        self.waiting_req_list = self.waiting_req_list[consumed_req_count:]
        return new_batch

    def _calcu_batch_token_load_batch_not_none(self, current_batch: Batch):

        estimated_peak_token_num = self._caclu_batch_estimated_peak_token_num(current_batch)

        return (estimated_peak_token_num, estimated_peak_token_num / self.max_total_tokens)
