import torch.multiprocessing as mp
from typing import List, Tuple, Optional
from lightllm.server.router.model_infer.infer_batch import InferReq
from lightllm.utils.log_utils import init_logger
from .prefill_impl import NIXLChunckedPrefillForPrefillNode, NIXLChunckedTransTask
from lightllm.server.router.model_infer.mode_backend.dp_backend.impl import DPChunkedPrefillBackend

logger = init_logger(__name__)


class NIXLDPChunkedForPrefillNode(DPChunkedPrefillBackend):
    def __init__(self, info_queue: mp.Queue, mem_queue: mp.Queue) -> None:
        super().__init__()
        self.support_overlap = False
        self.info_queue: mp.Queue = info_queue
        self.mem_queue: mp.Queue = mem_queue
        self.classed_req_no_decode = True
        self.nixl_prefill_chuncked_handle_func = self._prefill_chuncked_handle_func

    def init_custom(self):
        NIXLChunckedPrefillForPrefillNode.init_custom(self)
        return

    def _filter_not_ready_reqs(self, req_ids: List[int]) -> List[InferReq]:
        return NIXLChunckedPrefillForPrefillNode._filter_not_ready_reqs(self, req_ids)

    def _prefill_chuncked_handle_func(
        self, req_obj: InferReq, next_token_id: int, next_token_prob: float, output_len: int
    ):
        return NIXLChunckedPrefillForPrefillNode._prefill_chuncked_handle_func(
            self, req_obj=req_obj, next_token_id=next_token_id, next_token_prob=next_token_prob, output_len=output_len
        )

    def _create_nixl_trans_task(
        self, req_obj: InferReq, kv_start_index: int, kv_end_index: int
    ) -> NIXLChunckedTransTask:
        return NIXLChunckedPrefillForPrefillNode._create_nixl_trans_task(
            self, req_obj=req_obj, kv_start_index=kv_start_index, kv_end_index=kv_end_index
        )
