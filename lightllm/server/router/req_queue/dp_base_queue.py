import random
from typing import List
from ..batch import Batch, Req
from lightllm.server.router.req_queue.base_queue import BaseQueue
from lightllm.common.basemodel.infer_lock import g_router_lock
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class DpQueue:
    def __init__(self, args, router, base_queue_class, dp_size_in_node) -> None:
        self.dp_size_in_node = dp_size_in_node
        self.base_queue_class = base_queue_class
        self.pre_select_dp_index = self.dp_size_in_node - 1
        from lightllm.server.router.manager import RouterManager

        self.router: RouterManager = router
        self.inner_queues: List[BaseQueue] = [
            base_queue_class(args, router, dp_index, dp_size_in_node) for dp_index in range(self.dp_size_in_node)
        ]

        return

    def get_dp_queue(self, dp_index: int):
        assert dp_index < self.dp_size_in_node, "dp index out of range"
        return self.inner_queues[dp_index]

    def get_paused_req_num(self, dp_index: int = 0):
        return self.inner_queues[dp_index].get_paused_req_num()

    def get_wait_req_num(self):
        return sum(queue.get_wait_req_num() for queue in self.inner_queues)

    # @calculate_time(show=True, min_cost_ms=10)
    def generate_new_batch(self, current_batch: Batch, limit_router_queue_length: int = None):
        batches = [
            self.inner_queues[dp_index].generate_new_batch(current_batch) for dp_index in range(self.dp_size_in_node)
        ]
        return self._merge_batch(batches)

    def _merge_batch(self, dp_batches: List[Batch]):
        merged_batch: Batch = None
        for iter_batch in dp_batches:
            if merged_batch is not None:
                merged_batch.dp_merge(iter_batch)
            else:
                merged_batch = iter_batch
        return merged_batch

    def append(self, req: Req):
        suggested_dp_index = req.sample_params.suggested_dp_index
        if suggested_dp_index >= self.dp_size_in_node or suggested_dp_index < 0:
            logger.warning(f"input req {req.request_id} dp index {suggested_dp_index} is invalid")
            suggested_dp_index = self._get_suggest_dp_index()
            self.pre_select_dp_index = suggested_dp_index
            req.sample_params.suggested_dp_index = suggested_dp_index
            self.inner_queues[suggested_dp_index].append(req)
        else:
            self.inner_queues[suggested_dp_index].append(req)
        return

    def extend(self, req_group: List[Req]):
        # 同一个组的，要分配在同一个 dp 上，效率最高
        index = self._get_suggest_dp_index()
        for req in req_group:
            suggested_dp_index = req.sample_params.suggested_dp_index
            if suggested_dp_index >= self.dp_size_in_node or suggested_dp_index < 0:
                logger.warning(f"input req {req.request_id} dp index {suggested_dp_index} is invalid")
                self.pre_select_dp_index = index
                req.sample_params.suggested_dp_index = index
                self.inner_queues[index].append(req)
            else:
                self.inner_queues[suggested_dp_index].append(req)

        return

    def back_to_wait_list(self, req_list: List[Req]):
        raise NotImplementedError("not supported feature")

    def is_busy(self):
        return True

    def update_token_load(self, current_batch: Batch, force_update=False):
        if self.router.shared_token_load.need_update_dynamic_max_load() or force_update:
            for dp_index in range(self.dp_size_in_node):
                estimated_peak_token_count, dynamic_max_load = self.inner_queues[dp_index].calcu_batch_token_load(
                    current_batch
                )
                token_ratio1 = self.router.get_used_tokens(dp_index) / self.router.max_total_token_num
                with g_router_lock.obj:
                    self.router.shared_token_load.set_current_load(token_ratio1, dp_index)
                    self.router.shared_token_load.set_estimated_peak_token_count(estimated_peak_token_count, dp_index)
                    self.router.shared_token_load.set_dynamic_max_load(dynamic_max_load, dp_index)
        return

    def _get_suggest_dp_index(self):
        min_length = min(len(queue.waiting_req_list) for queue in self.inner_queues)
        select_dp_indexes = [
            i for i, queue in enumerate(self.inner_queues) if len(queue.waiting_req_list) == min_length
        ]

        # multi thread safe keep
        if not select_dp_indexes:
            return random.randint(0, self.dp_size_in_node - 1)

        # round_robin select.
        for i in range(self.dp_size_in_node):
            next_dp_index = (self.pre_select_dp_index + i + 1) % self.dp_size_in_node
            if next_dp_index in select_dp_indexes:
                return next_dp_index

        return random.choice(select_dp_indexes)
