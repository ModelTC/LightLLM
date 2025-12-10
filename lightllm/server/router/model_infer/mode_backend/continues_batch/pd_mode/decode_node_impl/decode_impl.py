import os
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import threading
from lightllm.server.router.model_infer.mode_backend.chunked_prefill.impl import ChunkedPrefillBackend
from typing import List, Tuple
from lightllm.server.router.model_infer.infer_batch import g_infer_context, InferReq, g_infer_state_lock
from lightllm.server.core.objs import FinishStatus
from lightllm.utils.log_utils import init_logger
from rpyc.utils.server import ThreadedServer
from lightllm.common.basemodel.infer_lock import g_router_lock
from .decode_task_cache import g_success_kv_move_task_cache, KVMoveTask
from lightllm.utils.envs_utils import get_unique_server_name
from lightllm.utils.dist_utils import create_new_group_for_current_dp

logger = init_logger(__name__)


class DecodeNode(ChunkedPrefillBackend):
    def __init__(self, info_queue: mp.Queue) -> None:
        super().__init__()
        self.info_queue: mp.Queue = info_queue
        self.classed_req_strict_prefill = False

    def init_custom(self):

        self.lock_nccl_group = create_new_group_for_current_dp("gloo")
        logger.info(f"lock_nccl_group ranks {dist.get_rank(self.lock_nccl_group)}")

        from .decode_infer_rpyc import PDDecodeInferRpcServer

        socket_path = f"/tmp/{get_unique_server_name()}_decode_node_infer_rpyc_{self.pd_rpyc_ports[self.rank_in_node]}"
        if os.path.exists(socket_path):
            os.remove(socket_path)

        t = ThreadedServer(
            PDDecodeInferRpcServer(self), socket_path=socket_path, protocol_config={"allow_pickle": True}
        )
        threading.Thread(target=lambda: t.start(), daemon=True).start()
        return

    def _init_reqs(self, reqs: List[Tuple]):
        """
        替换请求初始化操作，替换为 Decode 节点独有的一些特殊初始化流程
        """
        if self.dp_size_in_node != 1:
            dp_rank_in_node = self.dp_rank_in_node
            reqs = [req for req in reqs if req[3] == dp_rank_in_node]

        g_infer_state_lock.acquire()

        uninit_reqs = g_infer_context.add_reqs(reqs, init_prefix_cache=False)
        # 匹配radix cache，并更新一些资源的管理。
        self._post_init_reqs(uninit_reqs=uninit_reqs)

        g_infer_state_lock.release()
        req_ids = [e[0] for e in reqs]
        return req_ids

    def _post_init_reqs(self, uninit_reqs: List[InferReq]):
        """
        检查请求的 kv len 将可能有问题的请求立即结束掉
        """
        if len(uninit_reqs) == 0:
            return

        remove_count = 0
        estimated_peak_token_count = 0
        for req_obj in uninit_reqs:
            req_obj: InferReq = req_obj  # for easy typing
            request_id = req_obj.req_id
            if request_id in g_success_kv_move_task_cache:
                task, share_node, _ = g_success_kv_move_task_cache.pop(request_id)
                task: KVMoveTask = task  # for easy typing
                self.radix_cache.dec_node_ref_counter(share_node)
                req_all_len = len(task.input_tokens) + task.decode_node.max_new_tokens
                remove_count += req_all_len
                estimated_peak_token_count += req_all_len
                req_obj._match_radix_cache()
            else:
                # 对于不合法的请求，直接模拟将其finished掉
                req_obj.cur_output_len += 1
                req_obj.set_next_gen_token_id(0, 0.0, 1)
                req_obj.finish_status.set_status(FinishStatus.FINISHED_STOP)

                if self.is_master_in_dp:
                    req_obj.shm_req.shm_cur_kv_len = req_obj.cur_kv_len
                    req_obj.shm_req.shm_cur_output_len = req_obj.cur_output_len
                    req_obj.shm_req.finish_token_index = req_obj.get_cur_total_len() - 1
                    req_obj.shm_req.finish_status.set_status(FinishStatus.FINISHED_STOP)
                    req_obj.shm_req.candetoken_out_len = req_obj.cur_output_len

                    req_id = req_obj.shm_req.request_id
                    logger.error(f"req_id: {req_id} forced to finished, it not in g_success_kv_move_task_cache")

        if self.is_master_in_dp:
            with g_router_lock.obj:
                self.shared_token_load.add_frozened_token_count(-remove_count, self.dp_rank_in_node)
                self.shared_token_load.add_estimated_peak_token_count(estimated_peak_token_count, self.dp_rank_in_node)
        return
