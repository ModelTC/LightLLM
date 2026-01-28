import torch.multiprocessing as mp
from typing import List, Tuple
from lightllm.server.router.model_infer.infer_batch import InferReq
from lightllm.utils.log_utils import init_logger
from .prefill_impl import ChunckedPrefillForPrefillNode
from lightllm.server.router.model_infer.mode_backend.dp_backend.impl import DPChunkedPrefillBackend

logger = init_logger(__name__)


class DPChunkedForPrefillNode(DPChunkedPrefillBackend):
    def __init__(self, info_queue: mp.Queue) -> None:
        super().__init__()
        self.support_overlap = False
        self.info_queue: mp.Queue = info_queue
        self.classed_req_no_decode = True

    def init_custom(self):
        ChunckedPrefillForPrefillNode.init_custom(self)
        return

    def _pre_handle_finished_reqs(self, finished_reqs):
        self._prefill_req_frozen_tokens_and_put_to_kvmove_taskqueue(finished_reqs=finished_reqs)
        return

    def _prefill_req_frozen_tokens_and_put_to_kvmove_taskqueue(self, finished_reqs: List[InferReq]):
        ChunckedPrefillForPrefillNode._prefill_req_frozen_tokens_and_put_to_kvmove_taskqueue(
            self, finished_reqs=finished_reqs
        )
        return
