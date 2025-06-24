import threading
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from typing import List, Tuple
from lightllm.utils.infer_utils import calculate_time, mark_start, mark_end
from lightllm.server.router.model_infer.infer_batch import InferReq, g_infer_context
from lightllm.utils.log_utils import init_logger
from lightllm.server.router.model_infer.mode_backend.generic_pre_process import prepare_prefill_inputs
from lightllm.server.router.model_infer.mode_backend.generic_post_process import sample
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.server.router.model_infer.mode_backend.dp_backend.pre_process import padded_prepare_prefill_inputs

from .impl_for_pd_base import PDNIXLBackendBase
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
        self._filter_reqs(aborted_reqs + ok_finished_reqs)

        # if ok_finished_reqs:
        #     for req in ok_finished_reqs:
        #         self._transfer_kv_to_remote(req)
        #     self._filter_reqs(ok_finished_reqs)
        #     ok_finished_reqs.clear()

        current_dp_prefill_num = len(prefill_reqs)
        self.reduce_tensor.fill_(current_dp_prefill_num)
        dist.all_reduce(self.reduce_tensor, op=dist.ReduceOp.MAX, group=None, async_op=False)
        max_prefill_num = self.reduce_tensor.item()
        if max_prefill_num != 0:
            if not self.enable_prefill_microbatch_overlap:
                self.normal_prefill_reqs(prefill_reqs, max_prefill_num, uinit_reqs, ok_finished_reqs)
            else:
                self.overlap_prefill_reqs(prefill_reqs, max_prefill_num, uinit_reqs, ok_finished_reqs)

        self._overlap_req_init_and_filter(uninit_reqs=uinit_reqs, ok_finished_reqs=ok_finished_reqs, clear_list=True)
        return

    def normal_prefill_reqs(self, prefill_reqs: List[InferReq], max_prefill_num: int, uninit_reqs, ok_finished_reqs):

        kwargs, run_reqs, padded_req_num = padded_prepare_prefill_inputs(
            prefill_reqs, max_prefill_num, is_multimodal=self.is_multimodal
        )
        logits = self.model.forward(**kwargs)
        self._overlap_req_init_and_filter(uninit_reqs=uninit_reqs, ok_finished_reqs=ok_finished_reqs, clear_list=True)
        if len(run_reqs) != 0:
            logits = logits[0 : len(run_reqs), :]
            next_token_ids, next_token_probs = sample(logits, run_reqs, self.eos_id)
            next_token_ids = next_token_ids.detach().cpu().numpy()
            next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()
            self._post_handle(
                run_reqs,
                next_token_ids,
                next_token_logprobs,
                is_chuncked_mode=True,
                do_filter_finished_reqs=False,
                extra_post_req_handle_chunk_func=lambda req: self.prefill_post_handle_queue.put(req),
            )

    def overlap_prefill_reqs(self, prefill_reqs: List[InferReq], max_prefill_num: int, uninit_reqs, ok_finished_reqs):
        from lightllm.server.router.model_infer.mode_backend.dp_backend.pre_process import (
            padded_overlap_prepare_prefill_inputs,
        )

        (
            micro_batch,
            run_reqs,
            padded_req_num,
            micro_batch1,
            run_reqs1,
            padded_req_num1,
        ) = padded_overlap_prepare_prefill_inputs(prefill_reqs, max_prefill_num, is_multimodal=self.is_multimodal)
        logits, logits1 = self.model.microbatch_overlap_prefill(micro_batch, micro_batch1)
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
                all_run_reqs,
                next_token_ids,
                next_token_logprobs,
                is_chuncked_mode=True,
                do_filter_finished_reqs=False,
                extra_post_req_handle_chunk_func=lambda req: self.prefill_post_handle_queue.put(req),
            )
