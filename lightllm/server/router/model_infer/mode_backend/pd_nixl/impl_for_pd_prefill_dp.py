import torch
import torch.multiprocessing as mp
from lightllm.server.router.model_infer.infer_batch import g_infer_context
from lightllm.utils.log_utils import init_logger
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.server.router.model_infer.mode_backend.dp_backend.impl import DPChunkedPrefillBackend

from .impl_for_pd_prefill import PDNIXLBackendForPrefillNode

logger = init_logger(__name__)


class PDNIXLDPBackendForPrefillNode(PDNIXLBackendForPrefillNode):
    def __init__(self, transfer_task_queue: mp.Queue, transfer_done_queue: mp.Queue, nixl_meta_queue: mp.Queue) -> None:
        super().__init__(transfer_task_queue, transfer_done_queue, nixl_meta_queue)
        self.enable_prefill_microbatch_overlap = get_env_start_args().enable_prefill_microbatch_overlap

    def init_custom(self):
        super().init_custom()
        self.reduce_tensor = torch.tensor([0], dtype=torch.int32, device="cuda", requires_grad=False)
        return

    def decode(self):
        uinit_reqs, aborted_reqs, ok_finished_reqs, prefill_reqs, decode_reqs = self._get_classed_reqs(
            g_infer_context.infer_req_ids,
            no_decode=True,
        )

        ok_finished_reqs, aborted_reqs, _ = self._prefill_filter_reqs(ok_finished_reqs, aborted_reqs)

        assert len(uinit_reqs) == 0
        assert len(decode_reqs) == 0

        self._prefill_abort_remote(aborted_reqs)
        self._filter_reqs(aborted_reqs)

        # 进行 chuncked prefill
        dp_prefill_req_nums, max_prefill_num = self._dp_all_gather_prefill_req_num(prefill_reqs=prefill_reqs)
        if self.chunked_prefill_state.dp_need_prefill(prefill_reqs, decode_reqs, dp_prefill_req_nums, max_prefill_num):
            if not self.enable_prefill_microbatch_overlap:
                DPChunkedPrefillBackend.normal_prefill_reqs(
                    self,
                    prefill_reqs,
                    max_prefill_num,
                    uinit_reqs,
                    ok_finished_reqs,
                    extra_post_req_handle_func=self._handle_chunked_transfer,
                    call_post_handle_for_chunk=True,
                )
            else:
                DPChunkedPrefillBackend.overlap_prefill_reqs(
                    self,
                    prefill_reqs,
                    max_prefill_num,
                    uinit_reqs,
                    ok_finished_reqs,
                    extra_post_req_handle_func=self._handle_chunked_transfer,
                    call_post_handle_for_chunk=True,
                )

        self._overlap_req_init_and_filter(uninit_reqs=uinit_reqs, ok_finished_reqs=ok_finished_reqs, clear_list=True)
        return
