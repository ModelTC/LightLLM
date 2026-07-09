from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

import torch

from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
from lightllm.common.basemodel.triton_kernel.mtp_utils import gen_b_req_mtp_start_loc
from lightllm.server.router.model_infer.infer_batch import g_infer_context
from lightllm.server.router.model_infer.pin_mem_manager import g_pin_mem_manager

if TYPE_CHECKING:
    from lightllm.server.router.model_infer.speculative.planner import SpecDecodePlan
    from lightllm.server.router.model_infer.speculative.runtime import SpecRuntime


@dataclass
class SpecDecodeForwardState:
    model_input: ModelInput
    original_run_reqs: List
    plan: "SpecDecodePlan"
    selected_run_reqs_cpu: Optional[torch.Tensor]
    accepted_index_cpu: torch.Tensor
    mtp_accept_len_cpu: torch.Tensor
    next_token_ids_cpu: torch.Tensor
    next_token_logprobs_cpu: torch.Tensor
    verify_event: torch.cuda.Event
    sync_event: torch.cuda.Event
    additional_mem_indexes_cpu: Optional[torch.Tensor]
    schedule_probs_cpu: Optional[torch.Tensor]


@dataclass
class SpecDecodePostState:
    next_token_ids: torch.Tensor
    next_token_logprobs: torch.Tensor
    mtp_accept_len_cpu: torch.Tensor
    need_free_mem_indexes: torch.Tensor


class SpecDecodeRunner:
    def __init__(self, runtime: "SpecRuntime") -> None:
        self.runtime = runtime
        self.backend = runtime.backend

    def run_speculative_forward(
        self,
        *,
        model_input: ModelInput,
        model_output: ModelOutput,
        run_reqs: List,
        req_num: int,
        plan: "SpecDecodePlan",
        selected_run_reqs_cpu: Optional[torch.Tensor],
        next_token_ids: torch.Tensor,
        next_token_logprobs: torch.Tensor,
        copy_next_token_infos: Callable[
            [torch.Tensor, torch.Tensor],
            Tuple[torch.Tensor, torch.Tensor],
        ],
    ) -> SpecDecodeForwardState:
        runtime = self.runtime
        b_req_mtp_start_loc = gen_b_req_mtp_start_loc(model_input.b_mtp_index, num_reqs=req_num)
        verify_result = runtime.verify_target_tokens(
            new_next_token_ids=next_token_ids,
            b_req_idx=model_input.b_req_idx,
            b_req_mtp_start_loc=b_req_mtp_start_loc,
        )
        accepted_index = verify_result.accepted_index
        accepted_index_cpu = g_pin_mem_manager.async_copy_from_gpu_tensor(
            key="accepted_index",
            gpu_tensor=accepted_index,
        )

        verify_event = torch.cuda.Event(enable_timing=True)
        verify_event.record()

        proposal = runtime.propose_next(
            main_model_input=model_input,
            main_model_output=model_output,
            next_token_ids=next_token_ids,
            b_req_mtp_start_loc=b_req_mtp_start_loc,
            draft_step=plan.draft_step,
            verify_result=verify_result,
        )
        all_next_token_ids = runtime.pad_all_next_token_ids(
            token_ids=proposal.token_ids,
            draft_step=plan.draft_step,
        )
        all_next_token_probs = runtime.build_all_next_token_probs(
            next_token_logprobs=next_token_logprobs,
            proposal=proposal,
            draft_step=plan.draft_step,
        )
        schedule_probs_cpu = (
            g_pin_mem_manager.async_copy_from_gpu_tensor(
                key="mtp_schedule_probs",
                gpu_tensor=all_next_token_probs,
            )
            if all_next_token_probs is not None
            else None
        )

        runtime.scatter_next_tokens(
            b_req_mtp_start_loc=b_req_mtp_start_loc,
            all_next_token_ids=all_next_token_ids,
            b_req_idx=model_input.b_req_idx,
            mtp_accept_len=verify_result.accept_len,
            all_next_token_probs=all_next_token_probs,
        )

        next_token_ids_cpu, next_token_logprobs_cpu = copy_next_token_infos(next_token_ids, next_token_logprobs)

        g_infer_context.req_sampling_manager.update_reqs_out_token_counter_gpu(
            b_req_idx=model_input.b_req_idx,
            next_token_ids=next_token_ids,
            mask=accepted_index == 1,
        )

        mtp_accept_len_cpu = g_pin_mem_manager.async_copy_from_gpu_tensor(
            key="mtp_accept_len",
            gpu_tensor=verify_result.accept_len,
        )

        sync_event = torch.cuda.Event()
        sync_event.record()

        return SpecDecodeForwardState(
            model_input=model_input,
            original_run_reqs=run_reqs,
            plan=plan,
            selected_run_reqs_cpu=selected_run_reqs_cpu,
            accepted_index_cpu=accepted_index_cpu,
            mtp_accept_len_cpu=mtp_accept_len_cpu,
            next_token_ids_cpu=next_token_ids_cpu,
            next_token_logprobs_cpu=next_token_logprobs_cpu,
            verify_event=verify_event,
            sync_event=sync_event,
            additional_mem_indexes_cpu=proposal.extra_mem_indexes_cpu,
            schedule_probs_cpu=schedule_probs_cpu,
        )

    def resolve_pre_post_reqs(self, *, state: SpecDecodeForwardState, decode_reqs: List):
        if state.plan.skip_verify_sync:
            assert self.runtime.enable_dynamic_mtp, "skip_verify_sync should only be True when dynamic MTP is enabled"
            return decode_reqs, decode_reqs

        state.verify_event.synchronize()
        return self.runtime.build_decode_req_lists(
            original_run_reqs=state.original_run_reqs,
            selected_run_reqs_cpu=state.selected_run_reqs_cpu,
            accepted_index_cpu=state.accepted_index_cpu,
        )

    def finish_post(self, *, state: SpecDecodeForwardState, req_num: int, run_reqs: List) -> SpecDecodePostState:
        state.sync_event.synchronize()

        runtime = self.runtime
        if runtime.enable_dynamic_mtp:
            runtime.update_dynamic_accept_stats(
                req_num=req_num,
                run_reqs=run_reqs,
                accepted_index_cpu=state.accepted_index_cpu,
                dynamic_batch_size=state.plan.dynamic_batch_size,
            )
            runtime.update_dynamic_schedule_stats(
                req_num=req_num,
                schedule_probs_cpu=state.schedule_probs_cpu,
            )

        need_free_mem_indexes = runtime.build_decode_free_mem_indexes_cpu(
            model_input=state.model_input,
            selected_run_reqs_cpu=state.selected_run_reqs_cpu,
            accepted_index_cpu=state.accepted_index_cpu,
        )
        if state.additional_mem_indexes_cpu is not None:
            need_free_mem_indexes = torch.cat([need_free_mem_indexes, state.additional_mem_indexes_cpu], dim=0)

        select_mask = state.accepted_index_cpu.to(dtype=torch.bool)
        return SpecDecodePostState(
            next_token_ids=state.next_token_ids_cpu[select_mask],
            next_token_logprobs=state.next_token_logprobs_cpu[select_mask],
            mtp_accept_len_cpu=state.mtp_accept_len_cpu,
            need_free_mem_indexes=need_free_mem_indexes,
        )
