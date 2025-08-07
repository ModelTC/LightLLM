from types import MethodType
import torch.multiprocessing as mp
from lightllm.server.router.model_infer.infer_batch import g_infer_context
from lightllm.utils.log_utils import init_logger

from .impl_for_pd_base import PDNIXLBackendBaseDPChunked
from .impl_for_pd_prefill import PDNIXLBackendForPrefillNode

logger = init_logger(__name__)


class PDNIXLDPBackendForPrefillNode(PDNIXLBackendBaseDPChunked):
    def __init__(self, transfer_task_queue: mp.Queue, transfer_done_queue: mp.Queue, nixl_meta_queue: mp.Queue) -> None:
        self.init_custom = MethodType(PDNIXLBackendForPrefillNode.init_custom, self)
        super().__init__(transfer_task_queue, transfer_done_queue, nixl_meta_queue)

        self.support_overlap = False
        self.classed_req_no_decode = True
        self.call_post_handle_for_chunk = True
        self.extra_post_req_handle_func = self._handle_chunked_transfer

        self._pre_handle_finished_reqs = MethodType(PDNIXLBackendForPrefillNode._pre_handle_finished_reqs, self)
