import torch.multiprocessing as mp
from lightllm.server.router.model_infer.infer_batch import g_infer_context, InferReq
from lightllm.utils.log_utils import init_logger
from typing import List, Tuple
from lightllm.server.router.model_infer.mode_backend.dp_backend.impl import DPChunkedPrefillBackend
from .decode_impl import NIXLDecodeNode, NIXLChunckedTransTaskGroup

logger = init_logger(__name__)


class NIXLDPForDecodeNode(DPChunkedPrefillBackend):
    def __init__(self, info_queue: mp.Queue, mem_queue: mp.Queue) -> None:
        super().__init__()
        self.info_queue: mp.Queue = info_queue
        self.mem_queue: mp.Queue = mem_queue
        self.classed_req_strict_prefill = False
        return

    def init_custom(self):
        return NIXLDecodeNode.init_custom(self)

    def _init_reqs(self, reqs: List[Tuple]):
        return NIXLDecodeNode._init_reqs(self, reqs=reqs)

    def _post_init_reqs(self, uninit_reqs: List[InferReq]):
        return NIXLDecodeNode._post_init_reqs(self, uninit_reqs=uninit_reqs)

    def _filter_not_ready_reqs(self, req_ids: List[int]) -> List[InferReq]:
        return NIXLDecodeNode._filter_not_ready_reqs(self, req_ids=req_ids)

    def _decode_node_gen_trans_tasks(self, req_obj: InferReq):
        return NIXLDecodeNode._decode_node_gen_trans_tasks(self, req_obj=req_obj)

    def _create_nixl_trans_task(
        self,
        req_obj: InferReq,
        mem_indexes: List[int],
        kv_start_index: int,
        kv_end_index: int,
        group: NIXLChunckedTransTaskGroup,
    ):
        return NIXLDecodeNode._create_nixl_trans_task(
            self,
            req_obj=req_obj,
            mem_indexes=mem_indexes,
            kv_start_index=kv_start_index,
            kv_end_index=kv_end_index,
            group=group,
        )
