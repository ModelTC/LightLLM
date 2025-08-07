import os
import time
import torch.multiprocessing as mp
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Dict
from lightllm.server.router.model_infer.infer_batch import g_infer_context, InferReq
from lightllm.server.core.objs.req import PDNIXLChunkedPrefillReq
from lightllm.utils.log_utils import init_logger
from lightllm.server.multimodal_params import MultimodalParams

from .pd_remote_prefill_obj import (
    RemotePrefillTask,
    RemotePrefillServerInfo,
    RemotePrefillRequest,
    RemoteTransferStatusType,
)

from .impl_for_pd_base import PDNIXLBackendBaseChunked

logger = init_logger(__name__)


class PDNIXLBackendForDecodeNode(PDNIXLBackendBaseChunked):
    def __init__(self, prefill_task_queue: mp.Queue, prefill_done_queue: mp.Queue, nix_meta_queue: mp.Queue) -> None:
        super().__init__(prefill_task_queue, prefill_done_queue, nix_meta_queue)
        self.classed_req_strict_prefill = False
        self.support_overlap = True

    def init_custom(self):
        super(type(self), self).init_custom()
        self.wait_prefill_thread = threading.Thread(
            target=self._start_async_loop, args=(self._prefill_wait_loop_async,), daemon=True
        )
        max_workers = int(os.getenv("PD_NIXL_MOVE_PAGE_POOL_SIZE", 4))
        self.wait_move_page_pool = ThreadPoolExecutor(max_workers)
        self.wait_prefill_thread.start()
        return

    def _build_remote_prefill_task(self, index: int, kwargs: Dict, req: InferReq):
        prefill_node = req.shm_req.sample_params.move_kv_to_decode_node.to_dict()
        prefill_node_info = RemotePrefillServerInfo(
            perfill_server_id=prefill_node["node_id"],
            prefill_server_ip=prefill_node["ip"],
            prefill_server_port=prefill_node["rpyc_port"],
        )

        mem_indexes = kwargs.get("mem_indexes")
        b_start_loc = kwargs.get("b_start_loc")
        prefill_request = RemotePrefillRequest(
            prompt=req.shm_req.get_prompt_ids(),
            sampling_params=req.shm_req.sample_params,
            multimodal_params=MultimodalParams.from_dict(req.multimodal_params),
            local_cached_len=req.cur_kv_len,
            token_ids=mem_indexes[b_start_loc[index] : b_start_loc[index + 1]],
            page_ids=self.page_scheduer.borrow(),  # get page ids for this request, blocking when not enough pages
        )
        return RemotePrefillTask(server_info=prefill_node_info, prefill_request=prefill_request)

    def _trigger_remote_prefill(self, req_id: int, index: int, kwargs: Dict, req: InferReq):
        remote_prefill_task = self._build_remote_prefill_task(index, kwargs, req)
        self.request_to_page_ids[req_id] = remote_prefill_task.prefill_request.page_ids
        self.to_remote_queue.put(remote_prefill_task)

    def _pre_handle_finished_reqs(self, finished_reqs: List[InferReq]):
        new_finished_reqs = []
        for req in finished_reqs:
            if req.infer_aborted and req.in_prefill_or_transfer:
                # those are in progress, we will handle them later
                pass
            else:
                new_finished_reqs.append(req)

        finished_reqs = new_finished_reqs

    def _get_classed_reqs(
        self,
        req_ids: List[int] = None,
        no_decode: bool = False,
        strict_prefill: bool = False,
        recover_paused: bool = False,
    ):
        prefill_reqs, decode_reqs = super(type(self), self)._get_classed_reqs(
            req_ids, no_decode, strict_prefill, recover_paused
        )
        prefill_reqs, decode_reqs, failed_reqs, _ = self._decode_filter_reqs(prefill_reqs, decode_reqs)

        if failed_reqs:
            g_infer_context.filter_reqs(failed_reqs)

        if prefill_reqs:
            kwargs, run_reqs = self._prepare_remote_prefill_inputs(prefill_reqs)
            for idx, run_req in enumerate(run_reqs):
                run_req: InferReq = run_req
                shm_req: PDNIXLChunkedPrefillReq = run_req.shm_req
                # forward each req to remote prefill
                # since the token index are the same across TPs, we only need to trigger prefill on master
                if self.is_master_in_dp:
                    run_req.remote_prefill_start = time.time()
                    # since this function may blocking the calling thread, so we do it in a thread pool
                    self.wait_move_page_pool.submit(
                        self._trigger_remote_prefill, shm_req.group_req_id, idx, kwargs, run_req
                    )

                shm_req.set_pd_req_rank_state(
                    self.rank_in_dp, RemoteTransferStatusType.IN_PROGRESS.value
                )  # set in progress state
                run_req.in_prefill_or_transfer = True
                self.remote_prefilled_reqs[shm_req.group_req_id] = run_req

            prefill_reqs.clear()

        return prefill_reqs, decode_reqs
