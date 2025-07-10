import threading
import torch
import torch.multiprocessing as mp
from typing import List, Tuple
from lightllm.utils.infer_utils import calculate_time, mark_start, mark_end
from lightllm.server.router.model_infer.infer_batch import InferReq, g_infer_context
from lightllm.utils.log_utils import init_logger
from lightllm.server.router.model_infer.mode_backend.generic_pre_process import prepare_prefill_inputs
from lightllm.server.router.model_infer.mode_backend.generic_post_process import sample
from .impl_for_pd_base import PDNIXLBackendBase

logger = init_logger(__name__)


class PDNIXLBackendForPrefillNode(PDNIXLBackendBase):
    def __init__(self, transfer_task_queue: mp.Queue, transfer_done_queue: mp.Queue, nixl_meta_queue: mp.Queue) -> None:
        super().__init__(transfer_task_queue, transfer_done_queue, nixl_meta_queue)

    def init_custom(self):
        super().init_custom()
        self.handle_prefill_loop_thread = threading.Thread(target=self._start_async_loop,
                                                           args=(self._handle_prefill_loop,),
                                                           daemon=True)
        self.wait_transfer_loop_thread = threading.Thread(target=self._start_async_loop,
                                                          args=(self._wait_page_transfer_loop,),
                                                          daemon=True)
        self.handle_transfer_loop_thread = threading.Thread(target=self._start_async_loop,
                                                            args=(self._handle_transfer_loop,),
                                                            daemon=True)

        self.handle_prefill_loop_thread.start()
        self.handle_transfer_loop_thread.start()
        self.wait_transfer_loop_thread.start()
        return

    def prefill(self, reqs: List[Tuple]):
        self._init_reqs(reqs)
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
        self._filter_reqs(aborted_reqs + ok_finished_reqs)

        if prefill_reqs:
            kwargs, run_reqs = prepare_prefill_inputs(
                prefill_reqs, is_chuncked_mode=True, is_multimodal=self.is_multimodal
            )

            logits = self.model.forward(**kwargs)
            next_token_ids, next_token_probs = sample(logits, run_reqs, self.eos_id)
            next_token_ids = next_token_ids.detach().cpu().numpy()
            next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()


            self._post_handle(
                run_reqs,
                next_token_ids,
                next_token_logprobs,
                is_chuncked_mode=True,
                do_filter_finished_reqs=False,
                extra_post_req_handle_chunk_func=self._handle_chunked_transfer,
            )
        return
