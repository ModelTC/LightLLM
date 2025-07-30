import threading
import torch.multiprocessing as mp
from typing import List, Tuple
from lightllm.server.router.model_infer.infer_batch import g_infer_context, InferReq
from lightllm.server.core.objs.req import PDNIXLChunkedPrefillReq
from lightllm.utils.log_utils import init_logger

from .impl_for_pd_base import PDNIXLBackendBaseChunked
from .pd_remote_prefill_obj import RemoteTransferStatusType

logger = init_logger(__name__)


class PDNIXLBackendForPrefillNode(PDNIXLBackendBaseChunked):
    def __init__(self, transfer_task_queue: mp.Queue, transfer_done_queue: mp.Queue, nixl_meta_queue: mp.Queue) -> None:
        super().__init__(transfer_task_queue, transfer_done_queue, nixl_meta_queue)
        self.support_overlap = False
        self.classed_req_no_decode = True
        self.extra_post_req_handle_func = self._handle_chunked_transfer
        self.call_post_handle_for_chunk = True

    def init_custom(self):
        super(type(self), self).init_custom()
        self.handle_prefill_loop_thread = threading.Thread(
            target=self._start_async_loop, args=(self._handle_prefill_loop,), daemon=True
        )
        self.wait_transfer_loop_thread = threading.Thread(
            target=self._start_async_loop, args=(self._wait_page_transfer_loop,), daemon=True
        )
        self.handle_transfer_loop_thread = threading.Thread(
            target=self._start_async_loop, args=(self._handle_transfer_loop,), daemon=True
        )

        self.handle_prefill_loop_thread.start()
        self.handle_transfer_loop_thread.start()
        self.wait_transfer_loop_thread.start()
        return

    def _pre_handle_finished_reqs(self, finished_reqs: List[InferReq]):
        new_finished_reqs = []
        need_remote_aborted_reqs = []
        for req in finished_reqs:
            if req.in_prefill_or_transfer:
                if req.infer_aborted:
                    need_remote_aborted_reqs.append(req)
                    new_finished_reqs.append(req)
                else:
                    if req.infer_nixl_rpd:
                        shm_req: PDNIXLChunkedPrefillReq = req.shm_req
                        state = shm_req.get_pd_req_state()
                        if state == RemoteTransferStatusType.SUCCESS.value:  # success
                            req.in_prefill_or_transfer = False
                            new_finished_reqs.append(req)
                        elif state == RemoteTransferStatusType.FAILED.value:  # failure
                            need_remote_aborted_reqs.append(req)
                            req.in_prefill_or_transfer = False
                            new_finished_reqs.append(req)
                        else:
                            logger.warning(f"remote prefill request {shm_req.group_req_id} unexpected state {state}")
                    else:
                        pass
            else:
                new_finished_reqs.append(req)

        finished_reqs = new_finished_reqs

        self._prefill_abort_remote(need_remote_aborted_reqs)
