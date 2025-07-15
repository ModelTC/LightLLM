import torch
from typing import List, Tuple, Callable, Optional
from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend
from lightllm.common.basemodel.batch_objs import ModelOutput

from lightllm.server.router.model_infer.infer_batch import g_infer_context, InferReq
from lightllm.server.router.model_infer.mode_backend.generic_post_process import sample
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.server.router.model_infer.mode_backend.pre import padded_prepare_prefill_inputs
from lightllm.server.router.model_infer.mode_backend.pre import padded_overlap_prepare_prefill_inputs
from lightllm.server.router.model_infer.mode_backend.pre import padded_prepare_decode_inputs
from lightllm.server.router.model_infer.mode_backend.pre import padded_overlap_prepare_decode_inputs


class DPChunkedPrefillBackend(ModeBackend):
    def __init__(self) -> None:
        super().__init__()
        """
        DPChunkedPrefillBackend 是DP的chuncked prefill 的单机实现，并不是标准的chunked prefill
        实现，这个模式最佳使用方式需要和 PD 分离模式进行配合。
        """
        self.enable_decode_microbatch_overlap = get_env_start_args().enable_decode_microbatch_overlap
        self.enable_prefill_microbatch_overlap = get_env_start_args().enable_prefill_microbatch_overlap
        pass

    def prefill(self, reqs: List[Tuple]):
        self._init_reqs(reqs, init_req_obj=False)
        return

    def decode(self):
        uninit_reqs, aborted_reqs, ok_finished_reqs, prefill_reqs, decode_reqs = self._get_classed_reqs(
            g_infer_context.infer_req_ids
        )

        if aborted_reqs:
            g_infer_context.filter_reqs(aborted_reqs)

        dp_prefill_req_nums, max_prefill_num = self._dp_all_gather_prefill_req_num(prefill_reqs=prefill_reqs)
        if self.chunked_prefill_state.dp_need_prefill(prefill_reqs, decode_reqs, dp_prefill_req_nums, max_prefill_num):
            if not self.enable_prefill_microbatch_overlap:
                self.normal_prefill_reqs(prefill_reqs, max_prefill_num, uninit_reqs, ok_finished_reqs)
            else:
                self.overlap_prefill_reqs(prefill_reqs, max_prefill_num, uninit_reqs, ok_finished_reqs)

        max_decode_num = self._dp_all_reduce_decode_req_num(decode_reqs=decode_reqs)
        if max_decode_num != 0:
            if not self.enable_decode_microbatch_overlap:
                self.normal_decode(decode_reqs, max_decode_num, uninit_reqs, ok_finished_reqs)
            else:
                self.overlap_decode(decode_reqs, max_decode_num, uninit_reqs, ok_finished_reqs)

        self._overlap_req_init_and_filter(uninit_reqs=uninit_reqs, ok_finished_reqs=ok_finished_reqs, clear_list=True)
        return

    def normal_prefill_reqs(
        self,
        prefill_reqs: List[InferReq],
        max_prefill_num: int,
        uninit_reqs,
        ok_finished_reqs,
        extra_post_req_handle_func: Optional[Callable[[InferReq, int, float], None]] = None,
        call_post_handle_for_chunk: bool = False,
    ):
        model_input, run_reqs, padded_req_num = padded_prepare_prefill_inputs(
            prefill_reqs, is_multimodal=self.is_multimodal
        )
        model_output: ModelOutput = self.model.forward(model_input)
        logits = model_output.logits
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
                extra_post_req_handle_func=extra_post_req_handle_func,
                call_post_handle_for_chunk=call_post_handle_for_chunk,
            )
        return

    def normal_decode(self, decode_reqs: List[InferReq], max_decode_num: int, uninit_reqs, ok_finished_reqs):
        model_input, run_reqs, padded_req_num = padded_prepare_decode_inputs(
            decode_reqs, is_multimodal=self.is_multimodal
        )
        model_output: ModelOutput = self.model.forward(model_input)
        logits = model_output.logits

        self._overlap_req_init_and_filter(uninit_reqs=uninit_reqs, ok_finished_reqs=ok_finished_reqs, clear_list=True)

        if len(run_reqs) != 0:
            logits = logits[0 : len(run_reqs), :]
            next_token_ids, next_token_probs = sample(logits, run_reqs, self.eos_id)
            next_token_ids = next_token_ids.detach().cpu().numpy()
            next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()
            self._post_handle(
                run_reqs, next_token_ids, next_token_logprobs, is_chuncked_mode=False, do_filter_finished_reqs=False
            )
        logits = None

    def overlap_decode(self, decode_reqs: List[InferReq], max_decode_num: int, uninit_reqs, ok_finished_reqs):
        (
            micro_input,
            run_reqs,
            padded_req_num,
            micro_input1,
            run_reqs1,
            padded_req_num1,
        ) = padded_overlap_prepare_decode_inputs(decode_reqs, is_multimodal=self.is_multimodal)
        model_output, model_output1 = self.model.microbatch_overlap_decode(micro_input, micro_input1)
        logits = model_output.logits
        logits1 = model_output1.logits
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

    def overlap_prefill_reqs(
        self,
        prefill_reqs: List[InferReq],
        max_prefill_num: int,
        uninit_reqs,
        ok_finished_reqs,
        extra_post_req_handle_func: Optional[Callable[[InferReq, int, float], None]] = None,
        call_post_handle_for_chunk: bool = False,
    ):
        (
            micro_input,
            run_reqs,
            padded_req_num,
            micro_input1,
            run_reqs1,
            padded_req_num1,
        ) = padded_overlap_prepare_prefill_inputs(prefill_reqs, is_multimodal=self.is_multimodal)
        model_output, model_output1 = self.model.microbatch_overlap_prefill(micro_input, micro_input1)
        logits = model_output.logits
        logits1 = model_output1.logits
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
                extra_post_req_handle_func=extra_post_req_handle_func,
                call_post_handle_for_chunk=call_post_handle_for_chunk,
            )
        return
