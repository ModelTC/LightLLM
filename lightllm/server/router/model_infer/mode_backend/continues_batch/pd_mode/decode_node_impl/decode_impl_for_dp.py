import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from typing import List, Tuple
from lightllm.server.router.model_infer.infer_batch import g_infer_context, InferReq
from lightllm.utils.log_utils import init_logger
from lightllm.server.router.model_infer.mode_backend.pre import padded_prepare_prefill_inputs
from lightllm.utils.envs_utils import get_unique_server_name, get_env_start_args
from .decode_impl import ContinuesBatchBackendForDecodeNode
from lightllm.server.router.model_infer.mode_backend.dp_backend.impl import DPChunkedPrefillBackend

logger = init_logger(__name__)


class DPForDecodeNode(ContinuesBatchBackendForDecodeNode):
    def __init__(self, info_queue: mp.Queue, mem_queue: mp.Queue) -> None:
        super().__init__(info_queue, mem_queue)
        self.enable_decode_microbatch_overlap = get_env_start_args().enable_decode_microbatch_overlap
        return

    def prefill(self, reqs: List[Tuple]):
        self._init_reqs(reqs, init_req_obj=False)
        return

    def decode(self):
        uninit_reqs, aborted_reqs, ok_finished_reqs, prefill_reqs, decode_reqs = self._get_classed_reqs(
            g_infer_context.infer_req_ids
        )
        assert len(prefill_reqs) == 0

        self._filter_reqs(aborted_reqs)

        max_decode_num = self._dp_all_reduce_decode_req_num(decode_reqs=decode_reqs)
        if max_decode_num != 0:
            if not self.enable_decode_microbatch_overlap:
                DPChunkedPrefillBackend.normal_decode(self, decode_reqs, max_decode_num, uninit_reqs, ok_finished_reqs)
            else:
                DPChunkedPrefillBackend.overlap_decode(self, decode_reqs, max_decode_num, uninit_reqs, ok_finished_reqs)

        self._overlap_req_init_and_filter(uninit_reqs=uninit_reqs, ok_finished_reqs=ok_finished_reqs, clear_list=True)
        return
