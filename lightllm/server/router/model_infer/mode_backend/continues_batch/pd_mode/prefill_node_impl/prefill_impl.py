import os
import time
import threading
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from typing import List, Tuple
from lightllm.server.router.model_infer.infer_batch import InferReq
from lightllm.server.pd_io_struct import KVMoveTask, DecodeNodeInfo
from lightllm.utils.log_utils import init_logger
from lightllm.common.basemodel.infer_lock import g_router_lock, g_infer_state_lock
from rpyc.utils.server import ThreadedServer
from .prefill_task_cache import g_kv_move_task_cache
from lightllm.utils.device_utils import kv_trans_use_p2p
from lightllm.utils.envs_utils import get_unique_server_name
from lightllm.utils.dist_utils import create_new_group_for_current_dp
from lightllm.server.router.model_infer.mode_backend.chunked_prefill.impl import ChunkedPrefillBackend

logger = init_logger(__name__)


class ChunckedPrefillForPrefillNode(ChunkedPrefillBackend):
    def __init__(self, info_queue: mp.Queue, mem_queue: mp.Queue) -> None:
        super().__init__()
        self.support_overlap = False
        self.info_queue: mp.Queue = info_queue
        self.mem_queue: mp.Queue = mem_queue
        self.classed_req_no_decode = True

    def init_custom(self):

        self.lock_nccl_group = create_new_group_for_current_dp("gloo")
        logger.info(f"lock_nccl_group ranks {dist.get_rank(self.lock_nccl_group)}")

        from .prefill_infer_rpyc import PDPrefillInferRpcServer

        socket_path = f"/tmp/{get_unique_server_name()}_prefill_node_infer_rpyc_{self.pd_rpyc_ports[self.rank_in_node]}"
        if os.path.exists(socket_path):
            os.remove(socket_path)

        t = ThreadedServer(
            PDPrefillInferRpcServer(self), socket_path=socket_path, protocol_config={"allow_pickle": True}
        )
        threading.Thread(target=lambda: t.start(), daemon=True).start()

        if kv_trans_use_p2p():
            from ..p2p_fix import reduce_tensor

            mp.reductions.reduce_tensor.__code__ = reduce_tensor.__code__

        return

    def _pre_handle_finished_reqs(self, finished_reqs):
        self._prefill_req_frozen_tokens_and_put_to_kvmove_taskqueue(finished_reqs=finished_reqs)
        return

    def _prefill_req_frozen_tokens_and_put_to_kvmove_taskqueue(self, finished_reqs: List[InferReq]):
        if len(finished_reqs) == 0:
            return

        # 提前在radix cache中回收相关的信息，并添加引用进行锁定，方便传输进程传输kv。
        if self.is_master_in_dp:
            logger.info("prefill_req_handle_and_frozen_tokens")

        g_infer_state_lock.acquire()
        try:
            for req in finished_reqs:

                # 区分abort 和 正常结束的请求，正常结束的请求才发起kv传输任务。
                if not req.finish_status.is_finished():
                    continue

                req: InferReq = req
                key = req.get_input_token_ids()[0 : req.cur_kv_len]
                key = torch.tensor(key, dtype=torch.int64, device="cpu")
                value = self.model.req_manager.req_to_token_indexs[req.req_idx][: req.cur_kv_len].detach().cpu()
                prefix_len = self.radix_cache.insert(key, value)
                old_prefix_len = 0 if req.shared_kv_node is None else req.shared_kv_node.node_prefix_total_len
                self.model.mem_manager.free(
                    self.model.req_manager.req_to_token_indexs[req.req_idx][old_prefix_len:prefix_len]
                )
                if req.shared_kv_node is not None:
                    self.radix_cache.dec_node_ref_counter(req.shared_kv_node)
                    req.shared_kv_node = None

                req.cur_kv_len = 0
                req.shm_req.shm_cur_kv_len = 0

                if req.shm_req.sample_params.move_kv_to_decode_node.exists:
                    # 注意兼容纯tp 和 tp dp 混合模式的逻辑
                    if self.is_master_in_dp:
                        g_router_lock.acquire()
                        self.shared_token_load.add_frozened_token_count(len(key), self.dp_rank_in_node)
                        g_router_lock.release()

                    share_node, kv_len, value = self.radix_cache.match_prefix(key, update_refs=True)
                    assert len(key) == len(value)
                    # 将下面的请求放入到任务队列中, 注意要使用raidx cache 返回的value
                    decode_node_info = DecodeNodeInfo(**req.shm_req.sample_params.move_kv_to_decode_node.to_dict())
                    task = KVMoveTask(
                        group_request_id=req.shm_req.group_req_id,
                        input_tokens=key.tolist(),
                        prefill_token_indexes=value.tolist(),
                        decode_token_indexes=None,
                        prefill_node_id=self.args.pd_node_id,
                        decode_node=decode_node_info,
                        move_kv_len=None,
                        prefill_dp_index=self.dp_rank_in_node,
                        decode_dp_index=None,
                        pd_master_node_id=req.shm_req.sample_params.pd_master_node_id.get(),
                        mark_start_time=time.time(),
                    )
                    g_kv_move_task_cache[task.group_request_id] = (task, share_node)

                    # 注意兼容纯 tp 和 tp dp 混合模式的逻辑
                    if self.is_master_in_dp:
                        self.info_queue.put(task)
        except BaseException as e:
            logger.exception(str(e))
        g_infer_state_lock.release()
        if self.is_master_in_dp:
            logger.info("prefill_req_handle_and_frozen_tokens end")
        return
