from typing import List, Dict
from lightllm.utils.infer_utils import calculate_time
from ..batch import Batch, Req
from lightllm.server.core.objs import FinishStatus
from lightllm.common.basemodel.infer_lock import g_router_lock
from lightllm.utils.config_utils import get_fixed_kv_len


class BaseQueue:
    def __init__(self, args, router, dp_index, dp_size_in_node) -> None:
        self.args = args
        self.dp_index = dp_index
        self.dp_size_in_node = dp_size_in_node
        from lightllm.server.router.manager import RouterManager

        self.router: RouterManager = router
        # max_total_token_num - get_fixed_kv_len() 是为了减去被特定
        # 推理模式预先占用了部分token kv 资源，这会导致整体可用的kv 资源
        # 在极端情况下减少，在非特定模式下，get_fixed_kv_len() 返回的都是
        # 0， 不会有任何影响。
        self.max_total_tokens = args.max_total_token_num - get_fixed_kv_len()
        assert args.batch_max_tokens is not None
        self.batch_max_tokens = args.batch_max_tokens
        self.running_max_req_size = args.running_max_req_size  # Maximum number of concurrent requests
        self.waiting_req_list: List[Req] = []  # List of queued requests
        self.router_token_ratio = args.router_token_ratio  # ratio to determine whether the router is busy
        self.router_max_new_token_len = args.router_max_new_token_len

    def extend(self, req_group: List[Req]):
        for req in req_group:
            req.sample_params.suggested_dp_index = self.dp_index
        self.waiting_req_list.extend(req_group)
        return

    def get_wait_req_num(self):
        return len(self.waiting_req_list)

    def is_busy(self):
        # 计算当前所有的token使用量, 如果使用了dynamic prompt cache, 使用的token量中不包含，cache tree 中未被引用的数据。
        cur_all_used_tokens = self.router.get_used_tokens(self.dp_index)
        # 判断当前服务是否处于token使用率过高的状态，过高的情况下，调度要偏向保守
        cur_token_ratio = (
            cur_all_used_tokens + self.router.shared_token_load.get_frozened_token_count(self.dp_index)
        ) / self.max_total_tokens
        is_busy = cur_token_ratio >= self.router_token_ratio
        return is_busy

    def get_batch_dp_req_size(self, current_batch: Batch):
        if current_batch is None:
            return 0
        if self.dp_size_in_node == 1:
            return len(current_batch.reqs)

        return len([req for req in current_batch.reqs if req.sample_params.suggested_dp_index == self.dp_index])

    def generate_new_batch(self, current_batch: Batch):
        """
        args:
            current_batch: current batch
        return:
            new batch
        """
        raise NotImplementedError()

    def calcu_batch_token_load(self, current_batch: Batch):
        if current_batch is None:
            return 0, self.router.shared_token_load.get_frozened_token_count(self.dp_index) / self.max_total_tokens
        else:
            return self._calcu_batch_token_load_batch_not_none(current_batch)

    def _calcu_batch_token_load_batch_not_none(self, current_batch: Batch):
        raise NotImplementedError()

    def update_token_load(self, current_batch: Batch, force_update=False):
        if self.router.shared_token_load.need_update_dynamic_max_load() or force_update:
            estimated_peak_token_count, dynamic_max_load = self.calcu_batch_token_load(current_batch)
            token_ratio1 = self.router.get_used_tokens(self.dp_index) / self.router.max_total_token_num
            with g_router_lock.obj:
                self.router.shared_token_load.set_current_load(token_ratio1, self.dp_index)
                self.router.shared_token_load.set_estimated_peak_token_count(estimated_peak_token_count, self.dp_index)
                self.router.shared_token_load.set_dynamic_max_load(dynamic_max_load, self.dp_index)
        return
