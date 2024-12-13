import os
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import threading
from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend
from typing import List
from lightllm.utils.infer_utils import set_random_seed
from lightllm.utils.infer_utils import calculate_time, mark_start, mark_end
from lightllm.server.router.model_infer.infer_batch import InferBatch, InferReq, InferSamplingParams, requests_mapping
from lightllm.server.io_struct import ReqRunStatus, FinishStatus
from lightllm.server.pd_io_struct import UpKVStatus
from lightllm.utils.log_utils import init_logger
from ..pre_process import prepare_prefill_inputs, prepare_decode_inputs
from ..post_process import sample
from .up_status import UpStatusManager
from rpyc.utils.server import ThreadedServer
from lightllm.common.basemodel.infer_lock import g_infer_state_lock, g_router_lock
from .decode_task_cache import g_success_kv_move_task_cache

logger = init_logger(__name__)


class ContinuesBatchBackendForDecodeNode(ModeBackend):
    def __init__(self, info_queue: mp.Queue, mem_queue: mp.Queue) -> None:
        super().__init__()
        self.info_queue: mp.Queue = info_queue
        self.mem_queue: mp.Queue = mem_queue

    def init_custom(self):
        self.lock_nccl_group = dist.new_group(backend="gloo")
        from .decode_infer_rpyc import PDDecodeInferRpcServer

        socket_path = f"/tmp/decode_node_infer_rpyc_{self.pd_rpyc_port}"
        if os.path.exists(socket_path):
            os.remove(socket_path)

        t = ThreadedServer(
            PDDecodeInferRpcServer(self), socket_path=socket_path, protocol_config={"allow_pickle": True}
        )
        threading.Thread(target=lambda: t.start(), daemon=True).start()
        return

    @calculate_time(show=False, min_cost_ms=300)
    def prefill_batch(self, batch_id):
        """
        检查请求的 kv len 将可能有问题的请求立即结束掉
        """
        output_dict = {}
        batch: InferBatch = self.cache.pop(batch_id)

        g_infer_state_lock.acquire()
        remove_count = 0
        estimated_peak_token_count = 0
        for request_id in batch.request_ids:
            if request_id in g_success_kv_move_task_cache:
                task, share_node, _ = g_success_kv_move_task_cache.pop(request_id)
                self.radix_cache.dec_node_ref_counter(share_node)
                req_all_len = len(task.input_tokens) + task.decode_node.max_new_tokens
                remove_count += req_all_len
                estimated_peak_token_count += req_all_len
            else:
                # 对于不合法的请求，直接模拟将其finished掉
                req_obj: InferReq = requests_mapping[request_id]
                req_obj.finish_status = FinishStatus.FINISHED_STOP
                metadata = {
                    "id": 0,
                    "logprob": 0.0,
                }
                output_dict[req_obj.r_id] = (
                    req_obj.req_status,
                    req_obj.cur_kv_len,
                    req_obj.get_output_len(),
                    [(0, metadata)],
                    req_obj.finish_status.value,  # 转化为整数，避免传送大对象,
                    None,
                )
                if self.tp_rank < self.dp_size:
                    logger.error(
                        f"req_id: {req_obj.group_req_id} forced to finished, it not in g_success_kv_move_task_cache"
                    )

        if self.tp_rank < self.dp_size:
            with g_router_lock.obj:
                self.shared_token_load.add_frozened_token_count(-remove_count, self.tp_rank)
                self.shared_token_load.add_estimated_peak_token_count(estimated_peak_token_count, self.tp_rank)
        g_infer_state_lock.release()

        self.cache[batch.batch_id] = batch
        return output_dict

    @calculate_time(show=True, min_cost_ms=200)
    def decode_batch(self, batch_id):
        return self.forward(batch_id, is_prefill=False)

    def forward(self, batch_id, is_prefill):
        # special code for return all prompt_logprobs
        output_dict = {}
        batch: InferBatch = self.cache.pop(batch_id)
        if is_prefill:
            kwargs, run_reqs = prepare_prefill_inputs(batch, self.radix_cache, self.is_multimodal)
        else:
            kwargs, run_reqs = prepare_decode_inputs(batch, self.radix_cache)

        logits = self.model.forward(**kwargs)
        next_token_ids, next_token_probs = sample(logits, run_reqs, self.eos_id)
        next_token_ids = next_token_ids.detach().cpu().numpy()
        next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()

        for req_obj, next_token_id, next_token_logprob in zip(run_reqs, next_token_ids, next_token_logprobs):
            # prefill and decode is same
            req_obj: InferReq = req_obj
            req_obj.cur_kv_len = len(req_obj.input_token_ids)
            req_obj.input_token_ids.append(next_token_id)
            req_obj.out_token_id_count[next_token_id] += 1
            req_obj.update_finish_status(self.eos_id)

            metadata = {
                "id": int(next_token_id),
                "logprob": float(next_token_logprob),
            }
            output_dict[req_obj.r_id] = (
                req_obj.req_status,
                req_obj.cur_kv_len,
                req_obj.get_output_len(),
                [(int(next_token_id), metadata)],
                req_obj.finish_status.value,  # 转化为整数，避免传送大对象,
                None,
            )  # 请求状态， 当前占用的kv的长度， 当前输出token的数量， 输出的token的id和元信息列表， 是否推理结束的状态， 额外保留参数

        self.cache[batch.batch_id] = batch
        return output_dict
