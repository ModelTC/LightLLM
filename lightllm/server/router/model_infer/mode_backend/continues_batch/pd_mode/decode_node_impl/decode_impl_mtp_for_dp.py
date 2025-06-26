import torch.multiprocessing as mp
import torch.distributed as dist
from lightllm.server.router.model_infer.infer_batch import g_infer_context
from lightllm.utils.log_utils import init_logger
from .decode_impl_for_dp import DPForDecodeNode
from ....dp_backend.impl_mtp import DPChunkedPrefillWithMTPBackend

logger = init_logger(__name__)


class DPForMtpDecodeNode(DPForDecodeNode):
    def __init__(self, info_queue: mp.Queue, mem_queue: mp.Queue) -> None:
        super().__init__(info_queue, mem_queue)
        return

    def init_model(self, kvargs):
        super().init_model(kvargs)
        DPChunkedPrefillWithMTPBackend._init_mtp_draft_model(self, kvargs)
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
                DPChunkedPrefillWithMTPBackend.normal_mtp_decode(
                    self, decode_reqs, max_decode_num, uninit_reqs, ok_finished_reqs
                )
            else:
                DPChunkedPrefillWithMTPBackend.overlap_mtp_decode(
                    self, decode_reqs, max_decode_num, uninit_reqs, ok_finished_reqs
                )

        self._overlap_req_init_and_filter(uninit_reqs=uninit_reqs, ok_finished_reqs=ok_finished_reqs, clear_list=True)
        return
