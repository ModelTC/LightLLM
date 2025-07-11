import time
import torch.multiprocessing as mp
import threading
from concurrent.futures import ThreadPoolExecutor
from lightllm.server.router.model_infer.mode_backend.continues_batch.impl import ContinuesBatchBackend
from typing import List, Tuple, Dict
from lightllm.server.router.model_infer.infer_batch import g_infer_context, InferReq
from lightllm.server.core.objs.req import PDNIXLChunkedPrefillReq
from lightllm.utils.log_utils import init_logger
from lightllm.server.multimodal_params import MultimodalParams

from .pd_remote_prefill_obj import RemotePrefillTask, RemotePrefillServerInfo, RemotePrefillRequest, RemoteTransferStatusType

from .impl_for_pd_base import PDNIXLBackendBase

logger = init_logger(__name__)


class PDNIXLBackendForDecodeNode(PDNIXLBackendBase):
    def __init__(self, prefill_task_queue: mp.Queue, prefill_done_queue: mp.Queue, nix_meta_queue: mp.Queue) -> None:
        super().__init__(prefill_task_queue, prefill_done_queue, nix_meta_queue)

    def init_custom(self):
        super().init_custom()
        self.wait_prefill_thread = threading.Thread(target=self._start_async_loop,
                                                    args=(self._prefill_wait_loop_async,),
                                                    daemon=True)
        self.wait_move_page_pool = ThreadPoolExecutor(max_workers=4)
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
            page_ids=self.page_scheduer.borrow() # get page ids for this request, blocking when not enough pages
        )
        return RemotePrefillTask(server_info=prefill_node_info, prefill_request=prefill_request)

    def _trigger_remote_prefill(self, req_id: int, index: int, kwargs: Dict, req: InferReq):
        remote_prefill_task = self._build_remote_prefill_task(index, kwargs, req)
        self.request_to_page_ids[req_id] = remote_prefill_task.prefill_request.page_ids
        self.to_remote_queue.put(remote_prefill_task)

    def prefill(self, reqs: List[Tuple]):
        self._init_reqs(reqs, init_req_obj=False)
        return

    def decode(self):

        uninit_reqs, aborted_reqs, ok_finished_reqs, prefill_reqs, decode_reqs = self._get_classed_reqs(
            g_infer_context.infer_req_ids,
            no_decode=False,
        )
        # filter out remote prefilling reqs
        prefill_reqs, aborted_reqs, decode_reqs, _ = self._decode_filter_reqs(prefill_reqs, aborted_reqs, decode_reqs)

        self._filter_reqs(aborted_reqs)

        # allocate kv cache, do remote prefill
        if prefill_reqs:
            # TODO: we could allocate cache later after remote prefill done and get a signal from remote
            #       but it will have a risk to not have enough cache for this request.
            kwargs, run_reqs = self._prepare_remote_prefill_inputs(prefill_reqs)
            for idx, run_req in enumerate(run_reqs):
                run_req: InferReq = run_req
                shm_req: PDNIXLChunkedPrefillReq = run_req.shm_req
                # forward each req to remote prefill
                # since the token index are the same across TPs, we only need to trigger prefill on master
                if self.is_master_in_dp:
                    run_req.remote_prefill_start = time.time()
                    # since this function may blocking the calling thread, so we do it in a thread pool
                    self.wait_move_page_pool.submit(self._trigger_remote_prefill,
                                                    shm_req.group_req_id, idx, kwargs, run_req)

                shm_req.set_pd_req_rank_state(self.rank_in_dp, RemoteTransferStatusType.IN_PROGRESS.value)  # set in progress state
                run_req.in_prefill_or_transfer = True
                self.remote_prefilled_reqs[shm_req.group_req_id] = run_req

        if decode_reqs:
            ContinuesBatchBackend.normal_decode(
                self, decode_reqs=decode_reqs, uninit_reqs=uninit_reqs, ok_finished_reqs=ok_finished_reqs)

        self._overlap_req_init_and_filter(uninit_reqs=uninit_reqs, ok_finished_reqs=ok_finished_reqs, clear_list=True)
        return
