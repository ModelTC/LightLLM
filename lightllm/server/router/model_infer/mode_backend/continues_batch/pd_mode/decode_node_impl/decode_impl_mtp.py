import torch.multiprocessing as mp
from lightllm.server.router.model_infer.infer_batch import g_infer_context
from .decode_impl import ContinuesBatchBackendForDecodeNode
from ...impl_mtp import ContinuesBatchWithMTPBackend
from lightllm.utils.log_utils import init_logger


logger = init_logger(__name__)


class ContinuesBatchBackendForMtpDecodeNode(ContinuesBatchBackendForDecodeNode):
    def __init__(self, info_queue: mp.Queue, mem_queue: mp.Queue) -> None:
        super().__init__(info_queue=info_queue, mem_queue=mem_queue)

    def init_model(self, kvargs):
        super().init_model(kvargs)
        ContinuesBatchWithMTPBackend._init_mtp_draft_model(self, kvargs)
        return

    def decode(self):
        uninit_reqs, aborted_reqs, ok_finished_reqs, prefill_reqs, decode_reqs = self._get_classed_reqs(
            g_infer_context.infer_req_ids,
            no_decode=False,
        )
        # p d 分离模式下， decode 节点不可能存在需要prefill操作的请求
        assert len(prefill_reqs) == 0

        self._filter_reqs(aborted_reqs)

        if decode_reqs:
            ContinuesBatchWithMTPBackend.normal_mtp_decode(
                self, decode_reqs=decode_reqs, uninit_reqs=uninit_reqs, ok_finished_reqs=ok_finished_reqs
            )

        self._overlap_req_init_and_filter(uninit_reqs=uninit_reqs, ok_finished_reqs=ok_finished_reqs, clear_list=True)
        return
