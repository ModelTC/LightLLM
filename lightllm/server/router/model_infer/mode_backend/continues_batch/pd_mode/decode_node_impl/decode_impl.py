import os
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import threading
from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend
from typing import List, Tuple
from lightllm.utils.infer_utils import set_random_seed
from lightllm.utils.infer_utils import calculate_time, mark_start, mark_end
from lightllm.server.router.model_infer.infer_batch import g_infer_context, InferReq, InferSamplingParams
from lightllm.server.core.objs import FinishStatus
from lightllm.server.pd_io_struct import UpKVStatus
from lightllm.utils.log_utils import init_logger
from .pre_process import prepare_decode_inputs
from ...post_process import sample
from .up_status import UpStatusManager
from rpyc.utils.server import ThreadedServer
from lightllm.common.basemodel.infer_lock import g_infer_state_lock, g_router_lock
from .decode_task_cache import g_success_kv_move_task_cache, KVMoveTask
from lightllm.utils.device_utils import kv_trans_use_p2p

logger = init_logger(__name__)


class ContinuesBatchBackendForDecodeNode(ModeBackend):
    def __init__(self, info_queue: mp.Queue, mem_queue: mp.Queue) -> None:
        super().__init__()
        self.info_queue: mp.Queue = info_queue
        self.mem_queue: mp.Queue = mem_queue

    def init_custom(self):
        self.lock_nccl_group = dist.new_group(backend="gloo")
        from .decode_infer_rpyc import PDDecodeInferRpcServer

        socket_path = f"/tmp/decode_node_infer_rpyc_{self.pd_rpyc_ports[self.tp_rank]}"
        if os.path.exists(socket_path):
            os.remove(socket_path)

        t = ThreadedServer(
            PDDecodeInferRpcServer(self), socket_path=socket_path, protocol_config={"allow_pickle": True}
        )
        threading.Thread(target=lambda: t.start(), daemon=True).start()

        if kv_trans_use_p2p():
            from ..p2p_fix import reduce_tensor

            mp.reductions.reduce_tensor.__code__ = reduce_tensor.__code__

        return

    def prefill(self, reqs: List[Tuple]):
        # 当 dp_size 不等于 1 的时候，需要提前从发来的请求参数中
        # 剔除掉不需要处理的请求参数, deepseekv2 这种特殊的模型
        # 在 dp 模式下 tp_rank == dp_rank
        if self.dp_size != 1:
            cur_dp_index = self.tp_rank
            reqs = [req for req in reqs if req[3] == cur_dp_index]

        g_infer_state_lock.acquire()
        g_infer_context.add_reqs(reqs, init_req_obj=False)  # 请求对象进行延迟初始化
        g_infer_state_lock.release()
        return

    def decode(self):

        kwargs, uninit_reqs, finished_reqs, run_reqs = prepare_decode_inputs()

        if len(run_reqs) != 0:
            logits = self.model.forward(**kwargs)

        if len(uninit_reqs) != 0 or len(finished_reqs) != 0:
            # 利用推理的时间，延迟折叠下一个请求的初始化和退出操作
            with torch.cuda.stream(g_infer_context.get_overlap_stream()):
                g_infer_state_lock.acquire()
                self.filter_finished_reqs(finished_reqs)
                g_infer_state_lock.release()

                g_infer_state_lock.acquire()
                self.post_init(uninit_reqs)
                g_infer_state_lock.release()

            torch.cuda.current_stream().wait_stream(g_infer_context.get_overlap_stream())

        if len(run_reqs) != 0:

            next_token_ids, next_token_probs = sample(logits, run_reqs, self.eos_id)
            next_token_ids = next_token_ids.detach().cpu().numpy()
            next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()

            for req_obj, next_token_id, next_token_logprob in zip(run_reqs, next_token_ids, next_token_logprobs):
                # prefill and decode is same
                req_obj: InferReq = req_obj
                req_obj.cur_kv_len = req_obj.get_cur_total_len()

                req_obj.set_next_gen_token_id(next_token_id, next_token_logprob)
                req_obj.cur_output_len += 1

                req_obj.out_token_id_count[next_token_id] += 1
                req_obj.update_finish_status(self.eos_id)

                if self.tp_rank < self.dp_size:
                    # shm_cur_kv_len shm_cur_output_len 是 router 调度进程需要读的信息
                    # finish_token_index finish_status candetoken_out_len 是
                    # detokenization 进程需要的信息，注意这些变量的写入顺序避免异步协同问题。
                    req_obj.shm_req.shm_cur_kv_len = req_obj.cur_kv_len
                    req_obj.shm_req.shm_cur_output_len = req_obj.cur_output_len

                    if req_obj.finish_status.is_finished():
                        req_obj.shm_req.finish_token_index = req_obj.get_cur_total_len() - 1
                        req_obj.shm_req.finish_status = req_obj.finish_status

                    req_obj.shm_req.candetoken_out_len = req_obj.cur_output_len
        return

    def post_init(self, uninit_reqs: List[InferReq]):
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
                req_obj.init_all()
            else:
                # 对于不合法的请求，直接模拟将其finished掉
                req_obj.init_all()
                req_obj.set_next_gen_token_id(0, 0.0)
                req_obj.cur_output_len += 1

                if self.tp_rank < self.dp_size:
                    req_obj.shm_req.shm_cur_kv_len = req_obj.cur_kv_len
                    req_obj.shm_req.shm_cur_output_len = req_obj.cur_output_len
                    req_obj.shm_req.finish_token_index = req_obj.get_cur_total_len() - 1
                    req_obj.shm_req.finish_status.set_status(FinishStatus.FINISHED_STOP)
                    req_obj.shm_req.candetoken_out_len = req_obj.cur_output_len

                    req_id = req_obj.shm_req.request_id
                    logger.error(f"req_id: {req_id} forced to finished, it not in g_success_kv_move_task_cache")

        if self.tp_rank < self.dp_size:
            with g_router_lock.obj:
                self.shared_token_load.add_frozened_token_count(-remove_count, self.tp_rank)
                self.shared_token_load.add_estimated_peak_token_count(estimated_peak_token_count, self.tp_rank)
        return

    def filter_finished_reqs(self, finished_reqs: List[InferReq]):
        finished_req_ids = [req.shm_req.request_id for req in finished_reqs]
        g_infer_context.filter(finished_req_ids)
        return
