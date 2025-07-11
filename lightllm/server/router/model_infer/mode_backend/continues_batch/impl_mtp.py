import os
import torch
import numpy as np
from typing import List, Tuple
from transformers.configuration_utils import PretrainedConfig
from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend
from lightllm.server.router.model_infer.infer_batch import g_infer_context
from lightllm.server.router.model_infer.mode_backend.pre import (
    prepare_prefill_inputs,
    prepare_decode_inputs,
)
from lightllm.server.router.model_infer.mode_backend.mtp_pre_process import (
    prepare_mtp_prefill_inputs,
)
from lightllm.server.router.model_infer.mode_backend.generic_post_process import sample
from lightllm.server.router.model_infer.infer_batch import InferReq
from lightllm.utils.infer_utils import calculate_time, mark_start, mark_end
from lightllm.utils.log_utils import init_logger
from lightllm.models.deepseek_mtp.model import Deepseek3MTPModel
from lightllm.common.basemodel.infer_lock import g_infer_state_lock
from lightllm.common.basemodel.batch_objs import ModelOutput


logger = init_logger(__name__)


class ContinuesBatchWithMTPBackend(ModeBackend):
    def __init__(self) -> None:
        super().__init__()

    def decode(self):
        uninit_reqs, aborted_reqs, ok_finished_reqs, prefill_reqs, decode_reqs = self._get_classed_reqs(
            g_infer_context.infer_req_ids
        )

        if aborted_reqs:
            g_infer_context.filter_reqs(aborted_reqs)

        if prefill_reqs:
            self.normal_mtp_prefill_reqs(
                prefill_reqs=prefill_reqs, uninit_reqs=uninit_reqs, ok_finished_reqs=ok_finished_reqs
            )

        if decode_reqs:
            self.normal_mtp_decode(decode_reqs=decode_reqs, uninit_reqs=uninit_reqs, ok_finished_reqs=ok_finished_reqs)

        self._overlap_req_init_and_filter(uninit_reqs=uninit_reqs, ok_finished_reqs=ok_finished_reqs, clear_list=True)
        return

    def normal_mtp_prefill_reqs(
        self, prefill_reqs: List[InferReq], uninit_reqs: List[InferReq], ok_finished_reqs: List[InferReq]
    ):
        model_input, run_reqs = prepare_prefill_inputs(
            prefill_reqs, is_chuncked_mode=not self.disable_chunked_prefill, is_multimodal=self.is_multimodal
        )
        model_output = self.model.forward(model_input)

        self._overlap_req_init_and_filter(uninit_reqs=uninit_reqs, ok_finished_reqs=ok_finished_reqs, clear_list=True)

        next_token_ids_gpu, next_token_probs = sample(model_output.logits, run_reqs, self.eos_id)
        next_token_ids_cpu = next_token_ids_gpu.detach().cpu().numpy()
        next_token_logprobs_cpu = torch.log(next_token_probs).detach().cpu().numpy()

        self._post_handle(
            run_reqs,
            next_token_ids_cpu,
            next_token_logprobs_cpu,
            is_chuncked_mode=not self.disable_chunked_prefill,
            do_filter_finished_reqs=False,
        )

        # mtp kv fill
        draft_next_token_ids_gpu = next_token_ids_gpu
        draft_model_output = model_output
        draft_model_input = model_input
        # spec prefill: MTP, 这个地方只是为了填充draft model的 kv， 并不会使用生成的token_id。
        for draft_model_idx in range(self.mtp_step):
            draft_model_input = prepare_mtp_prefill_inputs(
                model_input=draft_model_input,
                b_next_token_ids=draft_next_token_ids_gpu,
                deepseekv3_mtp_draft_input_hiddens=draft_model_output.deepseekv3_mtp_main_output_hiddens,
            )

            draft_model_output = self.draft_models[draft_model_idx].forward(draft_model_input)
            draft_next_token_ids_gpu, draft_next_token_ids_cpu = self._gen_argmax_token_ids(draft_model_output)
        return

    def normal_mtp_decode(
        self, decode_reqs: List[InferReq], uninit_reqs: List[InferReq], ok_finished_reqs: List[InferReq]
    ):
        model_input, run_reqs = prepare_decode_inputs(decode_reqs)
        model_output = self.model.forward(model_input)

        self._overlap_req_init_and_filter(uninit_reqs=uninit_reqs, ok_finished_reqs=ok_finished_reqs, clear_list=True)

        next_token_ids_gpu, next_token_probs = sample(model_output.logits, run_reqs, self.eos_id)
        next_token_ids_cpu = next_token_ids_gpu.detach().cpu().numpy()
        next_token_logprobs_cpu = torch.log(next_token_probs).detach().cpu().numpy()

        # verify
        mem_indexes_cpu = model_input.mem_indexes.detach().cpu().numpy()
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

        # share some inference info with the main model
        draft_model_input = model_input
        draft_model_output = model_output
        draft_next_token_ids = next_token_ids_gpu
        # process the draft model output
        for draft_model_idx in range(self.mtp_step):

            draft_model_input.input_ids = draft_next_token_ids
            draft_model_input.deepseekv3_mtp_draft_input_hiddens = draft_model_output.deepseekv3_mtp_main_output_hiddens
            # spec decode: MTP
            draft_model_output: ModelOutput = self.draft_models[draft_model_idx].forward(draft_model_input)
            draft_next_token_ids, draft_next_token_ids_cpu = self._gen_argmax_token_ids(draft_model_output)

            unique_reqs = [run_reqs[index] for index in verify_ok_req_last_indexes]
            self._update_reqs_mtp_gen_token_ids(
                reqs=unique_reqs, mtp_draft_next_token_ids=draft_next_token_ids_cpu[verify_ok_req_last_indexes]
            )

        if need_free_mem_indexes:
            g_infer_state_lock.acquire()
            g_infer_context.req_manager.mem_manager.free(need_free_mem_indexes)
            g_infer_state_lock.release()
        return
