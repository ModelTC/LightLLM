from __future__ import annotations

from collections import Counter
from typing import List, Optional, Tuple

import torch

from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
from lightllm.common.basemodel.triton_kernel.mtp_utils import (
    linear_att_mtp_state_index_update,
    mtp_scatter_next_token_ids,
    mtp_verify,
)
from lightllm.server.router.model_infer.pin_mem_manager import AsyncPinnedCpuTensor, g_pin_mem_manager
from lightllm.server.router.model_infer.speculative.planner import (
    DSparkPlanner,
    FixedSpecPlanner,
    LightSpecPlanner,
    SpecDecodePlan,
)
from lightllm.server.router.model_infer.speculative.proposers import build_spec_proposer
from lightllm.server.router.model_infer.speculative.proposers.base import SpecProposal


class SpecEngine:
    """Owns speculative planning, verification, and proposal generation.

    The model backend controls target forward, sampling, stream synchronization,
    and request post-processing. Each proposer owns its algorithm-specific draft
    state and proposal generation.
    """

    def __init__(self, backend, spec_mode: str, enable_dynamic_spec: bool) -> None:
        self.backend = backend
        self.spec_mode = spec_mode
        self.enable_dynamic_spec = enable_dynamic_spec
        self.proposer = build_spec_proposer(
            spec_mode=spec_mode,
            backend=backend,
            enable_dynamic_spec=enable_dynamic_spec,
        )
        self.planner = self._build_decode_planner()
        self._register_cuda_graph_costs()

    # Prefill draft-state initialization.

    def build_draft_state_from_prefill(
        self,
        target_model_input: ModelInput,
        target_model_output: ModelOutput,
        next_token_ids: torch.Tensor,
    ) -> None:
        self.proposer.build_draft_state_from_prefill(
            target_model_input=target_model_input,
            target_model_output=target_model_output,
            next_token_ids=next_token_ids,
        )

    def build_draft_state_from_prefill_overlap(
        self,
        target_model_input0: ModelInput,
        target_model_output0: ModelOutput,
        next_token_ids0: torch.Tensor,
        target_model_input1: ModelInput,
        target_model_output1: ModelOutput,
        next_token_ids1: torch.Tensor,
    ) -> None:
        self.proposer.build_draft_state_from_prefill_overlap(
            target_model_input0=target_model_input0,
            target_model_output0=target_model_output0,
            next_token_ids0=next_token_ids0,
            target_model_input1=target_model_input1,
            target_model_output1=target_model_output1,
            next_token_ids1=next_token_ids1,
        )

    # Decode planning and target verification.

    def plan_decode(self, model_input: ModelInput, decode_reqs: List) -> SpecDecodePlan:
        """Return the fixed or dynamic speculative plan for one decode iteration."""

        req_num = len(decode_reqs)
        if isinstance(self.planner, LightSpecPlanner):
            # Prefill initializes draft cache but does not create a proposal.
            # After one decode, cur_output_len > 1 and the request owns the
            # previous iteration's candidates.
            proposal_req_num = sum(req.cur_output_len > 1 for req in decode_reqs)
            return self.planner.plan(
                req_num=req_num,
                original_batch_size=model_input.batch_size,
                proposal_req_num=proposal_req_num,
            )
        return self.planner.plan(req_num=req_num, original_batch_size=model_input.batch_size)

    def prepare_decode_model_input(
        self,
        model_input: ModelInput,
        req_num: int,
        plan: SpecDecodePlan,
    ) -> Tuple[ModelInput, Optional[AsyncPinnedCpuTensor]]:
        """Apply target verify-row compaction when the dynamic planner selects it."""

        if not plan.is_dynamic:
            return model_input, None

        if plan.dynamic_batch_size == model_input.batch_size:
            return model_input, None

        from lightllm.common.basemodel.triton_kernel.mtp_utils import prepare_dynamic_mtp_model_input

        model_input, selected_row_mask = prepare_dynamic_mtp_model_input(
            model_input=model_input,
            req_num=req_num,
            dynamic_batch_size=plan.dynamic_batch_size,
            req_to_next_token_scores=(
                self.backend.model.req_manager.req_sampling_params_manager.req_to_next_token_scores
            ),
            pre_draft_step=plan.pre_draft_step,
        )
        selected_row_mask_cpu = g_pin_mem_manager.async_copy_from_gpu_tensor_with_event(
            key="selected_row_mask",
            gpu_tensor=selected_row_mask,
        )
        return model_input, selected_row_mask_cpu

    def verify_tokens(
        self,
        next_token_ids: torch.Tensor,
        b_req_idx: torch.Tensor,
        b_req_mtp_start_loc: torch.Tensor,
        b_mtp_index: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        accept_lengths, accepted_index = mtp_verify(
            req_to_next_token_ids=self.backend.model.req_manager.req_sampling_params_manager.req_to_next_token_ids,
            b_req_mtp_start_loc=b_req_mtp_start_loc,
            new_next_token_ids=next_token_ids,
            b_req_idx=b_req_idx,
        )
        if self.backend.is_linear_att_mixed_model:
            assert b_mtp_index is not None
            linear_att_mtp_state_index_update(
                req_to_mtp_state_index=self.backend.model.req_manager.req_to_mtp_state_index,
                b_req_mtp_start_loc=b_req_mtp_start_loc,
                b_req_idx=b_req_idx,
                b_mtp_index=b_mtp_index,
                accepted_index=accepted_index,
                verify_width=self.backend.max_draft_step + 1,
            )
        return accept_lengths, accepted_index

    def resolve_decode_reqs(
        self,
        plan: SpecDecodePlan,
        verify_event: torch.cuda.Event,
        run_reqs: List,
        decode_reqs: List,
        accepted_index_cpu: torch.Tensor,
    ) -> Tuple[List, List]:
        """Return requests ready for pre/post handling after verification."""

        if plan.skip_verify_sync:
            return decode_reqs, decode_reqs

        verify_event.synchronize()
        verify_ok_reqs = [req for req, accepted in zip(run_reqs, accepted_index_cpu.tolist()) if accepted]
        return run_reqs, verify_ok_reqs

    # Draft proposal generation and persistence.

    def propose_next(
        self,
        main_model_input: ModelInput,
        main_model_output: ModelOutput,
        next_token_ids: torch.Tensor,
        b_req_mtp_start_loc: torch.Tensor,
        draft_step: int,
        accept_len: Optional[torch.Tensor] = None,
    ) -> SpecProposal:
        return self.proposer.propose_next(
            main_model_input=main_model_input,
            main_model_output=main_model_output,
            next_token_ids=next_token_ids,
            b_req_mtp_start_loc=b_req_mtp_start_loc,
            draft_step=draft_step,
            accept_len=accept_len,
        )

    def propose_next_overlap(
        self,
        main_model_input0: ModelInput,
        main_model_output0: ModelOutput,
        next_token_ids0: torch.Tensor,
        real_verify_rows0: int,
        accept_len0: torch.Tensor,
        main_model_input1: ModelInput,
        main_model_output1: ModelOutput,
        next_token_ids1: torch.Tensor,
        real_verify_rows1: int,
        accept_len1: torch.Tensor,
        draft_step: int,
    ) -> SpecProposal:
        # TODO: DP overlap currently disables dynamic drafting and always uses
        # max_draft_step. Share the score-feedback path with propose_next when
        # dynamic overlap is supported.
        return self.proposer.propose_next_overlap(
            main_model_input0=main_model_input0,
            main_model_output0=main_model_output0,
            next_token_ids0=next_token_ids0,
            real_verify_rows0=real_verify_rows0,
            accept_len0=accept_len0,
            main_model_input1=main_model_input1,
            main_model_output1=main_model_output1,
            next_token_ids1=next_token_ids1,
            real_verify_rows1=real_verify_rows1,
            accept_len1=accept_len1,
            draft_step=draft_step,
        )

    def scatter_next_tokens(
        self,
        b_req_mtp_start_loc: torch.Tensor,
        all_next_token_ids: torch.Tensor,
        b_req_idx: torch.Tensor,
        mtp_accept_len: torch.Tensor,
        schedule_scores: Optional[torch.Tensor] = None,
    ) -> None:
        mtp_scatter_next_token_ids(
            req_to_next_token_ids=self.backend.model.req_manager.req_sampling_params_manager.req_to_next_token_ids,
            b_req_mtp_start_loc=b_req_mtp_start_loc,
            all_next_token_ids=all_next_token_ids,
            b_req_idx=b_req_idx,
            mtp_accept_len=mtp_accept_len,
            req_to_next_token_scores=(
                self.backend.model.req_manager.req_sampling_params_manager.req_to_next_token_scores
                if schedule_scores is not None
                else None
            ),
            schedule_scores=schedule_scores,
        )

    # Planner feedback, request metrics, and resource cleanup.

    def update_planner_feedback(
        self,
        plan: SpecDecodePlan,
        proposal: SpecProposal,
        req_num: int,
        accept_lengths_cpu: torch.Tensor,
    ) -> None:
        """Feed iteration-level observations into the dynamic planner."""

        if not self.enable_dynamic_spec:
            return

        self.planner.update_feedback(
            plan=plan,
            req_num=req_num,
            accept_lengths=accept_lengths_cpu,
            schedule_scores=proposal.schedule_scores_cpu,
        )

    def record_request_spec_metrics(
        self,
        decode_reqs: List,
        accept_lengths_cpu: torch.Tensor,
        verified_row_reqs: Optional[List] = None,
    ) -> None:
        """Accumulate user-visible speculative metrics on each request."""

        if not self.backend.is_master_in_dp:
            return

        accept_lengths = accept_lengths_cpu.tolist()
        assert len(accept_lengths) == len(decode_reqs)
        verify_rows_by_req = None if verified_row_reqs is None else Counter(req.req_idx for req in verified_row_reqs)
        for req, accept_len in zip(decode_reqs, accept_lengths):
            req.update_spec_accepted_token_num(accept_token_num=accept_len - 1)
            verify_token_num = req.mtp_step + 1 if verify_rows_by_req is None else verify_rows_by_req[req.req_idx]
            if verify_token_num > 0:
                req.update_spec_verify_token_num(verify_token_num=verify_token_num)
                req.update_spec_verify_step_num(verify_step_num=1)

    def free_unused_decode_mem(
        self,
        model_input: ModelInput,
        selected_row_mask_cpu: Optional[torch.Tensor],
        accepted_index_cpu: torch.Tensor,
        extra_mem_indexes_cpu: Optional[torch.Tensor],
    ) -> None:
        """Free rejected target KV slots and draft-only temporary slots."""

        mem_indexes_cpu = model_input.mem_indexes_cpu
        if selected_row_mask_cpu is None:
            free_mask = accepted_index_cpu == 0
        else:
            selected_mask = selected_row_mask_cpu.to(dtype=torch.bool)
            free_mask = selected_mask.logical_not()
            free_mask[selected_mask] = accepted_index_cpu == 0
        need_free_mem_indexes = mem_indexes_cpu[free_mask]

        if extra_mem_indexes_cpu is not None:
            need_free_mem_indexes = torch.cat([need_free_mem_indexes, extra_mem_indexes_cpu], dim=0)
        if len(need_free_mem_indexes) > 0:
            self.backend.model.req_manager.mem_manager.free(need_free_mem_indexes)

    # Construction helpers.

    def _build_decode_planner(self):
        if not self.enable_dynamic_spec:
            return FixedSpecPlanner(max_draft_step=self.backend.max_draft_step)
        if self.spec_mode == "dspark":
            return DSparkPlanner(
                max_draft_step=self.backend.max_draft_step,
                draft_cost_provider=self.proposer,
            )

        return LightSpecPlanner(
            max_draft_step=self.backend.max_draft_step,
            draft_cost_provider=self.proposer,
        )

    def _register_cuda_graph_costs(self) -> None:
        if not self.enable_dynamic_spec:
            return

        target_graph = self.backend.model.graph
        if target_graph is not None:
            for batch_size, infer_cost_ms in target_graph.infer_cost_ms_by_batch_size.items():
                self.planner.update_infer_cost(
                    batch_size=batch_size,
                    infer_cost_ms=infer_cost_ms,
                    is_draft_model=False,
                )

        for draft_model in self.backend.draft_models:
            draft_graph = draft_model.graph
            if draft_graph is None:
                continue
            for batch_size, infer_cost_ms in draft_graph.infer_cost_ms_by_batch_size.items():
                self.planner.update_infer_cost(
                    batch_size=batch_size,
                    infer_cost_ms=infer_cost_ms,
                    is_draft_model=True,
                )
