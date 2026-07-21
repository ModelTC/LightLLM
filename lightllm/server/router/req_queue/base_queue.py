import time
from typing import List, Dict
from lightllm.utils.infer_utils import calculate_time
from ..batch import Batch, Req
from lightllm.server.core.objs import FinishStatus
from lightllm.utils.config_utils import get_fixed_kv_len
from lightllm.server.core.objs import StartArgs
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class BaseQueue:
    def __init__(self, args: StartArgs, router, dp_index, dp_size_in_node) -> None:
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

    def free_aborted_req_cpu_cache_pages(self, req: Req):
        if self.args.enable_cpu_cache:
            self.router.cpu_cache_client.lock.acquire_sleep1ms()
            self.router.cpu_cache_client.deref_pages(req.cpu_cache_match_page_indexes.get_all())
            req.cpu_cache_match_page_indexes.clear()
            self.router.cpu_cache_client.lock.release()

    def should_release_aborted_req_in_queue(self, req: Req):
        # 多节点 TP 的 waiting req abort 状态必须先由 rank 0 broadcast 对齐，
        # 不能在各节点本地调度队列里提前按各自 shm 状态释放。
        return req.is_aborted and not self.router.is_multinode_tp

    def mark_aborted_req_finished(self, req: Req):
        # 未开始推理的请求没有生成 token；这里写入一个 EOS 位置和 aborted 状态，
        # 让 httpserver recycle loop 能正常结束请求并返回空字符串。
        input_len = req.input_len
        req.link_prompt_ids_shm_array()
        req.link_logprobs_shm_array()
        req.finish_token_index = input_len
        req.shm_prompt_ids.arr[input_len] = self.args.eos_id[0]
        # shm_logprobs 为 structured array: [("logprob", f32), ("rank", i32)]
        req.shm_logprobs.arr["logprob"][input_len] = 0.0
        req.shm_logprobs.arr["rank"][input_len] = -1
        req.finish_status.set_status(FinishStatus.FINISHED_ABORTED)

        # 所有数据准备完后再通知 detokenizer
        req.candetoken_out_len = 1
        # 未进 Infer，无 final_token_metadata 可写；置位以免 HTTP 空等
        req.shm_infer_released = True

    def release_aborted_req(self, req: Req):
        logger.debug(f"router abort req id {req.request_id} shm_index: {req.index_in_shm_mem}")
        self.free_aborted_req_cpu_cache_pages(req)
        self.mark_aborted_req_finished(req)
        self.router.shm_req_manager.put_back_req_obj(req)
        return

    def filter_aborted_reqs(self):
        # 只释放 should_release_aborted_req_in_queue 为真的请求。
        # 采一波 → sleep 10ms → 再采；前后 request_id 集合完全一致才释放，
        # 避免同组 abort 标记写全前提前摘掉。多机 TP 下门禁为 False，不进入释放路径。
        aborted_reqs = [req for req in self.waiting_req_list if self.should_release_aborted_req_in_queue(req)]
        if not aborted_reqs:
            return

        prev_ids = {req.request_id for req in aborted_reqs}
        for _ in range(100):
            time.sleep(0.01)
            aborted_reqs = [req for req in self.waiting_req_list if self.should_release_aborted_req_in_queue(req)]
            cur_ids = {req.request_id for req in aborted_reqs}
            if prev_ids == cur_ids:
                break
            prev_ids = cur_ids
        else:
            # 100 次仍未稳定，本轮不释放，下轮调度再试
            logger.warning(
                f"aborted reqs not stable after 100 retries, skip release this round, "
                f"aborted_ids={sorted(prev_ids)}"
            )
            return

        aborted_ids = {req.request_id for req in aborted_reqs}
        self.waiting_req_list = [req for req in self.waiting_req_list if req.request_id not in aborted_ids]
        for req in aborted_reqs:
            self.release_aborted_req(req)
        return

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
        cur_token_ratio = cur_all_used_tokens / self.max_total_tokens
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
            return 0, 0.0
        else:
            return self._calcu_batch_token_load_batch_not_none(current_batch)

    def _calcu_batch_token_load_batch_not_none(self, current_batch: Batch):
        raise NotImplementedError()

    def update_token_load(self, current_batch: Batch, force_update=False):
        if self.router.shared_token_load.need_update_dynamic_max_load() or force_update:
            estimated_peak_token_count, dynamic_max_load = self.calcu_batch_token_load(current_batch)
            token_ratio1 = self.router.get_used_tokens(self.dp_index) / self.router.max_total_token_num
            self.router.shared_token_load.set_current_load(token_ratio1, self.dp_index)
            self.router.shared_token_load.set_estimated_peak_token_count(estimated_peak_token_count, self.dp_index)
            self.router.shared_token_load.set_dynamic_max_load(dynamic_max_load, self.dp_index)
        return
