import torch
import numpy as np
from typing import List, Tuple
from lightllm.server.router.model_infer.infer_batch import g_infer_context, InferReq
from lightllm.server.router.model_infer.mode_backend.generic_post_process import sample
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.server.router.model_infer.mode_backend.continues_batch.impl_mtp import ContinuesBatchWithMTPBackend
from lightllm.server.router.model_infer.mode_backend.pre import padded_prepare_prefill_inputs
from lightllm.server.router.model_infer.mode_backend.pre import padded_overlap_prepare_prefill_inputs
from lightllm.server.router.model_infer.mode_backend.pre import padded_prepare_decode_inputs
from lightllm.server.router.model_infer.mode_backend.pre import padded_overlap_prepare_decode_inputs
from lightllm.server.router.model_infer.mode_backend.mtp_pre_process import (
    prepare_mtp_prefill_inputs,
)
from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
from lightllm.common.basemodel.infer_lock import g_infer_state_lock


class DPChunkedPrefillWithMTPBackend(ContinuesBatchWithMTPBackend):
    def __init__(self) -> None:
        super().__init__()
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
                self.normal_mtp_prefill_reqs(prefill_reqs, max_prefill_num, uninit_reqs, ok_finished_reqs)
            else:
                self.overlap_mtp_prefill_reqs(prefill_reqs, max_prefill_num, uninit_reqs, ok_finished_reqs)

        max_decode_num = self._dp_all_reduce_decode_req_num(decode_reqs=decode_reqs)
        if max_decode_num != 0:
            if not self.enable_decode_microbatch_overlap:
                self.normal_mtp_decode(decode_reqs, max_decode_num, uninit_reqs, ok_finished_reqs)
            else:
                self.overlap_mtp_decode(decode_reqs, max_decode_num, uninit_reqs, ok_finished_reqs)

        self._overlap_req_init_and_filter(uninit_reqs=uninit_reqs, ok_finished_reqs=ok_finished_reqs, clear_list=True)
        return

    def normal_mtp_prefill_reqs(
        self, prefill_reqs: List[InferReq], max_prefill_num: int, uninit_reqs, ok_finished_reqs
    ):
        # main model prefill
        model_input, run_reqs, padded_req_num = padded_prepare_prefill_inputs(
            prefill_reqs, is_multimodal=self.is_multimodal
        )
        model_output: ModelOutput = self.model.forward(model_input)

        self._overlap_req_init_and_filter(uninit_reqs=uninit_reqs, ok_finished_reqs=ok_finished_reqs, clear_list=True)
        next_token_ids_cpu = []

        if len(run_reqs) != 0:
            next_token_ids_gpu, next_token_probs = sample(model_output.logits[: len(run_reqs)], run_reqs, self.eos_id)
            next_token_ids_cpu = next_token_ids_gpu.detach().cpu().numpy()
            next_token_logprobs_cpu = torch.log(next_token_probs).detach().cpu().numpy()

            self._post_handle(
                run_reqs,
                next_token_ids_cpu,
                next_token_logprobs_cpu,
                is_chuncked_mode=True,
                do_filter_finished_reqs=False,
            )

        # fill mtp draft model prefill kv
        # 因为存在padding的请求，需要将padding的请求一并考虑同时进行推理。
        draft_model_input = model_input
        draft_next_token_ids_gpu = torch.zeros((model_input.batch_size), dtype=torch.int64, device="cuda")
        if len(run_reqs) != 0:
            draft_next_token_ids_gpu[0 : len(run_reqs)].copy_(next_token_ids_gpu)

        draft_model_output = model_output

        for draft_model_idx in range(self.mtp_step):
            draft_model_input = prepare_mtp_prefill_inputs(
                model_input=draft_model_input,
                b_next_token_ids=draft_next_token_ids_gpu,
                deepseekv3_mtp_draft_input_hiddens=draft_model_output.deepseekv3_mtp_main_output_hiddens,
            )

            draft_model_output = self.draft_models[draft_model_idx].forward(draft_model_input)
            draft_next_token_ids_gpu, _ = self._gen_argmax_token_ids(draft_model_output)
        return

    def normal_mtp_decode(self, decode_reqs: List[InferReq], max_decode_num: int, uninit_reqs, ok_finished_reqs):
        model_input, run_reqs, padded_req_num = padded_prepare_decode_inputs(
            decode_reqs, is_multimodal=self.is_multimodal
        )
        # main model decode
        model_output = self.model.forward(model_input)

        self._overlap_req_init_and_filter(uninit_reqs=uninit_reqs, ok_finished_reqs=ok_finished_reqs, clear_list=True)

        need_free_mem_indexes = []
        verify_ok_req_last_indexes = []
        if len(run_reqs) != 0:
            next_token_ids_gpu, next_token_probs = sample(model_output.logits[: len(run_reqs)], run_reqs, self.eos_id)
            next_token_ids_cpu = next_token_ids_gpu.detach().cpu().numpy()
            next_token_logprobs_cpu = torch.log(next_token_probs).detach().cpu().numpy()

            # verify
            mem_indexes_cpu = model_input.mem_indexes[0 : len(run_reqs)].cpu().numpy()
            verify_ok_reqs, verify_ok_req_indexes, verify_ok_req_last_indexes, need_free_mem_indexes = self._verify_mtp(
                run_reqs, next_token_ids_cpu, mem_indexes_cpu
            )

            self._post_handle(
                verify_ok_reqs,
                next_token_ids_cpu[verify_ok_req_indexes],
                next_token_logprobs_cpu[verify_ok_req_indexes],
                is_chuncked_mode=False,
                do_filter_finished_reqs=False,
            )

        # fill draft model kv and gen next mtp token ids.
        draft_model_input = model_input
        draft_model_output = model_output
        draft_next_token_ids_gpu = torch.zeros((model_input.batch_size), dtype=torch.int64, device="cuda")
        if len(run_reqs) != 0:
            draft_next_token_ids_gpu[0 : len(run_reqs)].copy_(next_token_ids_gpu)

        # process the draft model output
        for draft_model_idx in range(self.mtp_step):

            draft_model_input.input_ids = draft_next_token_ids_gpu
            draft_model_input.deepseekv3_mtp_draft_input_hiddens = draft_model_output.deepseekv3_mtp_main_output_hiddens
            # spec decode: MTP
            draft_model_output: ModelOutput = self.draft_models[draft_model_idx].forward(draft_model_input)
            draft_next_token_ids_gpu, draft_next_token_ids_cpu = self._gen_argmax_token_ids(draft_model_output)

            if verify_ok_req_last_indexes:
                unique_reqs = [run_reqs[index] for index in verify_ok_req_last_indexes]
                self._update_reqs_mtp_gen_token_ids(
                    reqs=unique_reqs, mtp_draft_next_token_ids=draft_next_token_ids_cpu[verify_ok_req_last_indexes]
                )

        if need_free_mem_indexes:
            g_infer_state_lock.acquire()
            g_infer_context.req_manager.mem_manager.free(need_free_mem_indexes)
            g_infer_state_lock.release()
        return

    def overlap_mtp_decode(self, decode_reqs: List[InferReq], max_decode_num: int, uninit_reqs, ok_finished_reqs):
        (
            micro_input,
            run_reqs,
            padded_req_num,
            micro_input1,
            run_reqs1,
            padded_req_num1,
        ) = padded_overlap_prepare_decode_inputs(decode_reqs, is_multimodal=self.is_multimodal)
        micro_output, micro_output1 = self.model.microbatch_overlap_decode(micro_input, micro_input1)

        self._overlap_req_init_and_filter(uninit_reqs=uninit_reqs, ok_finished_reqs=ok_finished_reqs, clear_list=True)

        req_num, req_num1 = len(run_reqs), len(run_reqs1)
        all_run_reqs = run_reqs + run_reqs1
        need_free_mem_indexes = []
        verify_ok_req_last_indexes = []
        if len(all_run_reqs) != 0:
            all_logits = torch.empty(
                (req_num + req_num1, micro_output.logits.shape[1]),
                dtype=micro_output.logits.dtype,
                device=micro_output.logits.device,
            )

            all_logits[0:req_num, :].copy_(micro_output.logits[0:req_num, :], non_blocking=True)
            all_logits[req_num : (req_num + req_num1), :].copy_(micro_output1.logits[0:req_num1, :], non_blocking=True)

            next_token_ids_gpu, next_token_probs = sample(all_logits, all_run_reqs, self.eos_id)
            next_token_ids_cpu = next_token_ids_gpu.detach().cpu().numpy()
            next_token_logprobs_cpu = torch.log(next_token_probs).detach().cpu().numpy()
            micro_mem_indexes_cpu = micro_input.mem_indexes[0:req_num].cpu()
            micro_mem_indexes_cpu1 = micro_input1.mem_indexes[0:req_num1].cpu()
            mem_indexes_cpu = torch.cat((micro_mem_indexes_cpu, micro_mem_indexes_cpu1), dim=0).numpy()

            # verify
            verify_ok_reqs, verify_ok_req_indexes, verify_ok_req_last_indexes, need_free_mem_indexes = self._verify_mtp(
                all_run_reqs, next_token_ids_cpu, mem_indexes_cpu
            )

            self._post_handle(
                verify_ok_reqs,
                next_token_ids_cpu[verify_ok_req_indexes],
                next_token_logprobs_cpu[verify_ok_req_indexes],
                is_chuncked_mode=False,
                do_filter_finished_reqs=False,
            )

        # share some inference info with the main model
        draft_micro_input, draft_micro_input1 = micro_input, micro_input1

        draft_next_token_ids_gpu = torch.zeros((micro_input.batch_size), dtype=torch.int64, device="cuda")
        draft_next_token_ids_gpu1 = torch.zeros((micro_input1.batch_size), dtype=torch.int64, device="cuda")
        if req_num > 0:
            draft_next_token_ids_gpu[0:req_num].copy_(next_token_ids_gpu[0:req_num])
        if req_num1 > 1:
            draft_next_token_ids_gpu1[0:req_num1].copy_(next_token_ids_gpu[req_num : (req_num + req_num1)])
        draft_micro_output, draft_micro_output1 = micro_output, micro_output1

        # process the draft model output
        for draft_model_idx in range(self.mtp_step):

            draft_micro_input.input_ids = draft_next_token_ids_gpu
            draft_micro_input.deepseekv3_mtp_draft_input_hiddens = draft_micro_output.deepseekv3_mtp_main_output_hiddens
            draft_micro_input1.input_ids = draft_next_token_ids_gpu1
            draft_micro_input1.deepseekv3_mtp_draft_input_hiddens = (
                draft_micro_output1.deepseekv3_mtp_main_output_hiddens
            )

            draft_micro_output, draft_micro_output1 = self.draft_models[draft_model_idx].microbatch_overlap_decode(
                draft_micro_input, draft_micro_input1
            )

            draft_next_token_ids_gpu, draft_next_token_ids_cpu = self._gen_argmax_token_ids(draft_micro_output)
            draft_next_token_ids_gpu1, draft_next_token_ids_cpu1 = self._gen_argmax_token_ids(draft_micro_output1)

            if verify_ok_req_last_indexes:
                all_draft_next_token_ids_cpu = np.concatenate(
                    [draft_next_token_ids_cpu[0:req_num], draft_next_token_ids_cpu1[0:req_num1]], axis=0
                )
                unique_reqs = [all_run_reqs[index] for index in verify_ok_req_last_indexes]
                self._update_reqs_mtp_gen_token_ids(
                    reqs=unique_reqs, mtp_draft_next_token_ids=all_draft_next_token_ids_cpu[verify_ok_req_last_indexes]
                )

        if need_free_mem_indexes:
            g_infer_state_lock.acquire()
            g_infer_context.req_manager.mem_manager.free(need_free_mem_indexes)
            g_infer_state_lock.release()
        return

    def overlap_mtp_prefill_reqs(
        self, prefill_reqs: List[InferReq], max_prefill_num: int, uninit_reqs, ok_finished_reqs
    ):
        (
            micro_input,
            run_reqs,
            padded_req_num,
            micro_input1,
            run_reqs1,
            padded_req_num1,
        ) = padded_overlap_prepare_prefill_inputs(prefill_reqs, is_multimodal=self.is_multimodal)

        micro_output, micro_output1 = self.model.microbatch_overlap_prefill(micro_input, micro_input1)

        self._overlap_req_init_and_filter(uninit_reqs=uninit_reqs, ok_finished_reqs=ok_finished_reqs, clear_list=True)

        req_num, req_num1 = len(run_reqs), len(run_reqs1)
        all_run_reqs = run_reqs + run_reqs1
        next_token_ids_cpu = []
        if len(all_run_reqs) != 0:
            all_logits = torch.empty(
                (len(all_run_reqs), micro_output.logits.shape[1]),
                dtype=micro_output.logits.dtype,
                device=micro_output.logits.device,
            )

            all_logits[0:req_num, :].copy_(micro_output.logits[0:req_num, :], non_blocking=True)
            all_logits[req_num : (req_num + req_num1), :].copy_(micro_output1.logits[0:req_num1, :], non_blocking=True)

            next_token_ids_gpu, next_token_probs = sample(all_logits, all_run_reqs, self.eos_id)
            next_token_ids_cpu = next_token_ids_gpu.detach().cpu().numpy()
            next_token_logprobs_cpu = torch.log(next_token_probs).detach().cpu().numpy()

            self._post_handle(
                all_run_reqs,
                next_token_ids_cpu,
                next_token_logprobs_cpu,
                is_chuncked_mode=True,
                do_filter_finished_reqs=False,
            )

        # spec prefill: MTP
        draft_micro_input, draft_micro_input1 = micro_input, micro_input1
        draft_next_token_ids_gpu = torch.zeros((micro_input.batch_size), dtype=torch.int64, device="cuda")
        if req_num > 0:
            draft_next_token_ids_gpu[0:req_num].copy_(next_token_ids_gpu[0:req_num])

        draft_next_token_ids_gpu1 = torch.zeros((micro_input1.batch_size), dtype=torch.int64, device="cuda")
        if req_num1 > 0:
            draft_next_token_ids_gpu1[0:req_num1].copy_(next_token_ids_gpu[req_num : (req_num + req_num1)])

        draft_micro_output, draft_micro_output1 = micro_output, micro_output1

        for draft_model_idx in range(self.mtp_step):

            draft_micro_input = prepare_mtp_prefill_inputs(
                model_input=draft_micro_input,
                b_next_token_ids=draft_next_token_ids_gpu,
                deepseekv3_mtp_draft_input_hiddens=draft_micro_output.deepseekv3_mtp_main_output_hiddens,
            )

            draft_micro_input1 = prepare_mtp_prefill_inputs(
                model_input=draft_micro_input1,
                b_next_token_ids=draft_next_token_ids_gpu1,
                deepseekv3_mtp_draft_input_hiddens=draft_micro_output1.deepseekv3_mtp_main_output_hiddens,
            )

            draft_micro_output, draft_micro_output1 = self.draft_models[draft_model_idx].microbatch_overlap_prefill(
                draft_micro_input, draft_micro_input1
            )
            draft_next_token_ids_gpu, _ = self._gen_argmax_token_ids(draft_micro_output)
            draft_next_token_ids_gpu1, _ = self._gen_argmax_token_ids(draft_micro_output1)
        return
