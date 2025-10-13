import random
from typing import List
from ..batch import Batch, Req
from lightllm.server.router.req_queue.base_queue import BaseQueue
from lightllm.server.router.req_queue.dp_balancer import get_dp_balancer
from lightllm.common.basemodel.infer_lock import g_router_lock
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class DpQueue:
    def __init__(self, args, router, base_queue_class, dp_size_in_node) -> None:
        self.dp_size_in_node = dp_size_in_node
        self.base_queue_class = base_queue_class
        from lightllm.server.router.manager import RouterManager

        self.router: RouterManager = router
        self.inner_queues: List[BaseQueue] = [
            base_queue_class(args, router, dp_index, dp_size_in_node) for dp_index in range(self.dp_size_in_node)
        ]
        # 在调度这放松，在推理时约束。
        # 避免prefill 模式下的情况下，推理完成了，调度没及时获取信息，导致调度bs 过小
        for queue in self.inner_queues:
            queue.batch_max_tokens = int(args.batch_max_tokens * 2)
        self.dp_balancer = get_dp_balancer(args, dp_size_in_node, self.inner_queues)
        self.reqs_waiting_for_dp_index: List[List[Req]] = []
        return

    def get_dp_queue(self, dp_index: int):
        assert dp_index < self.dp_size_in_node, "dp index out of range"
        return self.inner_queues[dp_index]

    def get_wait_req_num(self):
        return sum(queue.get_wait_req_num() for queue in self.inner_queues)

    # @calculate_time(show=True, min_cost_ms=10)
    def generate_new_batch(self, current_batch: Batch):
        self.dp_balancer.assign_reqs_to_dp(current_batch, self.reqs_waiting_for_dp_index)
        batches = [
            self.inner_queues[dp_index].generate_new_batch(current_batch) for dp_index in range(self.dp_size_in_node)
        ]
        return self._merge_batch(batches)

    def _merge_batch(self, dp_batches: List[Batch]):
        merged_batch: Batch = None
        for iter_batch in dp_batches:
            if merged_batch is not None:
                merged_batch.merge(iter_batch)
            else:
                merged_batch = iter_batch
        return merged_batch

    def extend(self, req_group: List[Req]):
        suggested_dp_index = req_group[0].sample_params.suggested_dp_index
        if suggested_dp_index >= self.dp_size_in_node or suggested_dp_index < 0:
            # 同一个组的，要分配在同一个 dp 上
            self.reqs_waiting_for_dp_index.append(req_group)
        else:
            self.inner_queues[suggested_dp_index].extend(req_group)
        return

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
