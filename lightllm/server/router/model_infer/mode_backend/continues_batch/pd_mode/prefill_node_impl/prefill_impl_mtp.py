import torch.multiprocessing as mp
from lightllm.server.router.model_infer.infer_batch import g_infer_context
from .prefill_impl import ChunckedPrefillForPrefillNode
from ...impl_mtp import ContinuesBatchWithMTPBackend
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class ChunckedPrefillForMtpPrefillNode(ChunckedPrefillForPrefillNode):
    def __init__(self, info_queue: mp.Queue, mem_queue: mp.Queue) -> None:
        super().__init__(info_queue=info_queue, mem_queue=mem_queue)
        return

    def init_model(self, kvargs):
        super().init_model(kvargs)
        ContinuesBatchWithMTPBackend._init_mtp_draft_model(self, kvargs)
        return

    def decode(self):
        uinit_reqs, aborted_reqs, ok_finished_reqs, prefill_reqs, decode_reqs = self._get_classed_reqs(
            g_infer_context.infer_req_ids,
            no_decode=True,
        )
        assert len(uinit_reqs) == 0
        assert len(decode_reqs) == 0

        self._filter_reqs(aborted_reqs)

        if ok_finished_reqs:
            self.prefill_req_frozen_tokens_and_put_to_kvmove_taskqueue(ok_finished_reqs)
            self._filter_reqs(ok_finished_reqs)
            ok_finished_reqs.clear()

        if prefill_reqs:
            ContinuesBatchWithMTPBackend.normal_mtp_prefill_reqs(
                self, prefill_reqs=prefill_reqs, uninit_reqs=uinit_reqs, ok_finished_reqs=ok_finished_reqs
            )
        return
