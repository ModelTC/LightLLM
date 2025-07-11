import time
import torch
import torch.multiprocessing as mp
from lightllm.server.router.model_infer.infer_batch import g_infer_context, InferReq
from lightllm.server.core.objs.req import PDNIXLChunkedPrefillReq
from lightllm.utils.log_utils import init_logger
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.server.router.model_infer.mode_backend.dp_backend.impl import DPChunkedPrefillBackend

from .impl_for_pd_decode import PDNIXLBackendForDecodeNode, RemoteTransferStatusType

logger = init_logger(__name__)


class PDNIXLDPBackendForDecodeNode(PDNIXLBackendForDecodeNode):
    def __init__(self, prefill_task_queue: mp.Queue, prefill_done_queue: mp.Queue, nix_meta_queue: mp.Queue) -> None:
        super().__init__(prefill_task_queue, prefill_done_queue, nix_meta_queue)
        self.enable_decode_microbatch_overlap = get_env_start_args().enable_decode_microbatch_overlap

    def init_custom(self):
        super().init_custom()

        self.reduce_tensor = torch.tensor([0], dtype=torch.int32, device="cuda", requires_grad=False)
        from lightllm.server.router.model_infer.mode_backend.pre import padded_prepare_prefill_inputs

        kwargs, run_reqs, padded_req_num = padded_prepare_prefill_inputs([], 1, is_multimodal=self.is_multimodal)
        self.model.forward(**kwargs)
        assert len(run_reqs) == 0 and padded_req_num == 1

        return

    def decode(self):

        uninit_reqs, aborted_reqs, ok_finished_reqs, prefill_reqs, decode_reqs = self._get_classed_reqs(
            g_infer_context.infer_req_ids,
            no_decode=False,
        )
        # filter out remote prefilling reqs
        prefill_reqs, aborted_reqs, decode_reqs, _ = self._decode_filter_reqs(prefill_reqs, aborted_reqs, decode_reqs)

        self._filter_reqs(aborted_reqs)

        # allocate kv cache, do remote prefill
        if prefill_reqs:
            # TODO: we could allocate cache later after remote prefill done and get a signal from remote
            #       but it will have a risk to not have enough cache for this request.
            kwargs, run_reqs = self._prepare_remote_prefill_inputs(prefill_reqs)
            for idx, run_req in enumerate(run_reqs):
                run_req: InferReq = run_req
                shm_req: PDNIXLChunkedPrefillReq = run_req.shm_req
                # forward each req to remote prefill
                # since the token index are the same across TPs, we only need to trigger prefill on master
                if self.is_master_in_dp:
                    run_req.remote_prefill_start = time.time()
                    # since this function may blocking the calling thread, so we do it in a thread pool
                    self.wait_move_page_pool.submit(self._trigger_remote_prefill,
                                                    shm_req.group_req_id, idx, kwargs, run_req)

                shm_req.set_pd_req_rank_state(self.rank_in_dp, RemoteTransferStatusType.IN_PROGRESS.value)  # set in progress state
                run_req.in_prefill_or_transfer = True
                self.remote_prefilled_reqs[shm_req.group_req_id] = run_req

        max_decode_num = self._dp_all_reduce_decode_req_num(decode_reqs=decode_reqs)
        if max_decode_num != 0:
            if not self.enable_decode_microbatch_overlap:
                DPChunkedPrefillBackend.normal_decode(self, decode_reqs, max_decode_num, uninit_reqs, ok_finished_reqs)
            else:
                DPChunkedPrefillBackend.overlap_decode(self, decode_reqs, max_decode_num, uninit_reqs, ok_finished_reqs)

        self._overlap_req_init_and_filter(uninit_reqs=uninit_reqs, ok_finished_reqs=ok_finished_reqs, clear_list=True)
        return

