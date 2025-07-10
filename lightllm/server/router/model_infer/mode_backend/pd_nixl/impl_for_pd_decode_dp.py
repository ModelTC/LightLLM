import time
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from typing import List
from lightllm.server.router.model_infer.infer_batch import g_infer_context, InferReq
from lightllm.server.core.objs.req import PDNIXLChunkedPrefillReq
from lightllm.utils.log_utils import init_logger
from lightllm.server.router.model_infer.mode_backend.generic_post_process import sample
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.server.router.model_infer.mode_backend.dp_backend.pre_process import padded_prepare_decode_inputs

from .impl_for_pd_decode import PDNIXLBackendForDecodeNode, RemoteTransferStatusType

logger = init_logger(__name__)


class PDNIXLDPBackendForDecodeNode(PDNIXLBackendForDecodeNode):
    def __init__(self, prefill_task_queue: mp.Queue, prefill_done_queue: mp.Queue, nix_meta_queue: mp.Queue) -> None:
        super().__init__(prefill_task_queue, prefill_done_queue, nix_meta_queue)
        self.enable_decode_microbatch_overlap = get_env_start_args().enable_decode_microbatch_overlap

    def init_custom(self):
        super().init_custom()

        self.reduce_tensor = torch.tensor([0], dtype=torch.int32, device="cuda", requires_grad=False)
        from lightllm.server.router.model_infer.mode_backend.dp_backend.pre_process import padded_prepare_prefill_inputs

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

        self.reduce_tensor.fill_(len(decode_reqs))
        dist.all_reduce(self.reduce_tensor, op=dist.ReduceOp.MAX)
        max_decode_num = self.reduce_tensor.item()
        if max_decode_num != 0:
            if not self.enable_decode_microbatch_overlap:
                self.normal_decode(decode_reqs, max_decode_num, uninit_reqs, ok_finished_reqs)
            else:
                self.overlap_decode(decode_reqs, max_decode_num, uninit_reqs, ok_finished_reqs)
        self._overlap_req_init_and_filter(uninit_reqs=uninit_reqs, ok_finished_reqs=ok_finished_reqs, clear_list=True)
        return

    def normal_decode(self, decode_reqs: List[InferReq], max_decode_num: int, uninit_reqs, ok_finished_reqs):

        kwargs, run_reqs, padded_req_num = padded_prepare_decode_inputs(
            decode_reqs, max_decode_num, is_multimodal=self.is_multimodal
        )
        logits = self.model.forward(**kwargs)
        self._overlap_req_init_and_filter(uninit_reqs=uninit_reqs, ok_finished_reqs=ok_finished_reqs, clear_list=True)
        if len(run_reqs) != 0:
            logits = logits[0 : len(run_reqs), :]
            next_token_ids, next_token_probs = sample(logits, run_reqs, self.eos_id)
            next_token_ids = next_token_ids.detach().cpu().numpy()
            next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()
            self._post_handle(
                run_reqs, next_token_ids, next_token_logprobs, is_chuncked_mode=False, do_filter_finished_reqs=False
            )
        return

    def overlap_decode(self, decode_reqs: List[InferReq], max_decode_num: int, uninit_reqs, ok_finished_reqs):
        from lightllm.server.router.model_infer.mode_backend.dp_backend.pre_process import (
            padded_overlap_prepare_decode_inputs,
        )

        (
            micro_batch,
            run_reqs,
            padded_req_num,
            micro_batch1,
            run_reqs1,
            padded_req_num1,
        ) = padded_overlap_prepare_decode_inputs(decode_reqs, max_decode_num, is_multimodal=self.is_multimodal)

        logits, logits1 = self.model.microbatch_overlap_decode(micro_batch, micro_batch1)
        self._overlap_req_init_and_filter(uninit_reqs=uninit_reqs, ok_finished_reqs=ok_finished_reqs, clear_list=True)
        req_num, req_num1 = len(run_reqs), len(run_reqs1)
        all_logits = torch.empty((req_num + req_num1, logits.shape[1]), dtype=logits.dtype, device=logits.device)

        all_logits[0:req_num, :].copy_(logits[0:req_num, :], non_blocking=True)
        all_logits[req_num : (req_num + req_num1), :].copy_(logits1[0:req_num1, :], non_blocking=True)

        all_run_reqs = run_reqs + run_reqs1
        if all_run_reqs:
            next_token_ids, next_token_probs = sample(all_logits, all_run_reqs, self.eos_id)
            next_token_ids = next_token_ids.detach().cpu().numpy()
            next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()
            self._post_handle(
                all_run_reqs, next_token_ids, next_token_logprobs, is_chuncked_mode=False, do_filter_finished_reqs=False
            )
        return
