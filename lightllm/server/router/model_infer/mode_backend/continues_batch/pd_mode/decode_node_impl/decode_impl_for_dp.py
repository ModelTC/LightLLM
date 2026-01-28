import torch.multiprocessing as mp
from lightllm.server.router.model_infer.infer_batch import g_infer_context, InferReq
from lightllm.utils.log_utils import init_logger
from typing import List, Tuple
from lightllm.server.router.model_infer.mode_backend.dp_backend.impl import DPChunkedPrefillBackend
from .decode_impl import DecodeNode

logger = init_logger(__name__)


class DPForDecodeNode(DPChunkedPrefillBackend):
    def __init__(self, info_queue: mp.Queue) -> None:
        super().__init__()
        self.info_queue: mp.Queue = info_queue
        self.classed_req_strict_prefill = False
        return

    def init_custom(self):
        DecodeNode.init_custom(self)
        return

    def _init_reqs(self, reqs: List[Tuple]):
        DecodeNode._init_reqs(self, reqs=reqs)
        return

    def _post_init_reqs(self, uninit_reqs: List[InferReq]):
        DecodeNode._post_init_reqs(self, uninit_reqs=uninit_reqs)
        return
