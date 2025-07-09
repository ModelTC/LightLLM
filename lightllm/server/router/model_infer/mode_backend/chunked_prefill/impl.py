from typing import List, Tuple
from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend
from lightllm.server.router.model_infer.mode_backend.continues_batch.impl import ContinuesBatchBackend
from lightllm.utils.log_utils import init_logger
from lightllm.server.router.model_infer.infer_batch import g_infer_context


logger = init_logger(__name__)


class ChunkedPrefillBackend(ModeBackend):
    def __init__(self) -> None:
        super().__init__()

    def decode(self):
        uninit_reqs, aborted_reqs, ok_finished_reqs, prefill_reqs, decode_reqs = self._get_classed_reqs(
            g_infer_context.infer_req_ids
        )

        if aborted_reqs:
            g_infer_context.filter_reqs(aborted_reqs)

        # 先 decode
        if decode_reqs:
            ContinuesBatchBackend.normal_decode(
                self, decode_reqs=decode_reqs, uninit_reqs=uninit_reqs, ok_finished_reqs=ok_finished_reqs
            )

        # 再 prefill
        if self.chunked_prefill_state.need_prefill(prefill_reqs=prefill_reqs, decode_reqs=decode_reqs):
            ContinuesBatchBackend.normal_prefill_reqs(
                self, prefill_reqs=prefill_reqs, uninit_reqs=uninit_reqs, ok_finished_reqs=ok_finished_reqs
            )

        self._overlap_req_init_and_filter(uninit_reqs=uninit_reqs, ok_finished_reqs=ok_finished_reqs, clear_list=True)
        return
