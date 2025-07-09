import torch
from typing import List, Tuple, Callable, Optional
from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend
from lightllm.server.router.model_infer.infer_batch import g_infer_context, InferReq
from lightllm.server.router.model_infer.mode_backend.pre import (
    prepare_prefill_inputs,
    prepare_decode_inputs,
)
from lightllm.server.router.model_infer.mode_backend.generic_post_process import sample
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class ContinuesBatchBackend(ModeBackend):
    def __init__(self) -> None:
        super().__init__()

    def decode(self):
        uninit_reqs, aborted_reqs, ok_finished_reqs, prefill_reqs, decode_reqs = self._get_classed_reqs(
            g_infer_context.infer_req_ids
        )

        if aborted_reqs:
            g_infer_context.filter_reqs(aborted_reqs)

        if prefill_reqs:
            self.normal_prefill_reqs(
                prefill_reqs=prefill_reqs, uninit_reqs=uninit_reqs, ok_finished_reqs=ok_finished_reqs
            )

        if decode_reqs:
            self.normal_decode(decode_reqs=decode_reqs, uninit_reqs=uninit_reqs, ok_finished_reqs=ok_finished_reqs)

        self._overlap_req_init_and_filter(uninit_reqs=uninit_reqs, ok_finished_reqs=ok_finished_reqs, clear_list=True)
        return

    def normal_prefill_reqs(
        self,
        prefill_reqs: List[InferReq],
        uninit_reqs: List[InferReq],
        ok_finished_reqs: List[InferReq],
        mask_func: Optional[Callable[[List[InferReq], torch.Tensor], None]] = None,
        extra_post_req_handle_func: Optional[Callable[[InferReq, int, float], None]] = None,
    ):
        model_input, run_reqs = prepare_prefill_inputs(
            prefill_reqs, is_chuncked_mode=not self.disable_chunked_prefill, is_multimodal=self.is_multimodal
        )
        model_output = self.model.forward(model_input)
        logits = model_output.logits

        self._overlap_req_init_and_filter(uninit_reqs=uninit_reqs, ok_finished_reqs=ok_finished_reqs, clear_list=True)

        if mask_func is not None:
            mask_func(run_reqs, logits)

        next_token_ids, next_token_probs = sample(logits, run_reqs, self.eos_id)
        next_token_ids = next_token_ids.detach().cpu().numpy()
        next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()

        self._post_handle(
            run_reqs,
            next_token_ids,
            next_token_logprobs,
            is_chuncked_mode=not self.disable_chunked_prefill,
            do_filter_finished_reqs=False,
            extra_post_req_handle_func=extra_post_req_handle_func,
        )
        return

    def normal_decode(
        self,
        decode_reqs: List[InferReq],
        uninit_reqs: List[InferReq],
        ok_finished_reqs: List[InferReq],
        mask_func: Optional[Callable[[List[InferReq], torch.Tensor], None]] = None,
        extra_post_req_handle_func: Optional[Callable[[InferReq, int, float], None]] = None,
    ):
        model_input, run_reqs = prepare_decode_inputs(decode_reqs)
        model_output = self.model.forward(model_input)
        logits = model_output.logits

        self._overlap_req_init_and_filter(uninit_reqs=uninit_reqs, ok_finished_reqs=ok_finished_reqs, clear_list=True)

        if mask_func is not None:
            mask_func(run_reqs, logits)

        next_token_ids, next_token_probs = sample(logits, run_reqs, self.eos_id)
        next_token_ids = next_token_ids.detach().cpu().numpy()
        next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()

        self._post_handle(
            run_reqs,
            next_token_ids,
            next_token_logprobs,
            is_chuncked_mode=False,
            do_filter_finished_reqs=False,
            extra_post_req_handle_func=extra_post_req_handle_func,
        )
        return
