from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
from lightllm.server.router.model_infer.pin_mem_manager import AsyncPinnedCpuTensor, g_pin_mem_manager
from lightllm.server.router.model_infer.mtp_speculative.planner import (
    BaseMtpPlanner,
    DSparkPlanner,
    FixedSpecPlanner,
    LightSpecPlanner,
    SpecDecodePlan,
)
from lightllm.server.router.model_infer.mtp_speculative.proposers import build_spec_proposer
from lightllm.server.router.model_infer.mtp_speculative.proposers.base import BaseSpecProposer, SpecProposal

if TYPE_CHECKING:
    from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend


class SpecEngine:
    """Owns non-DP MTP planning and draft proposal generation.

    Target verification, request metrics, stream synchronization, and resource
    cleanup are stateless operations exposed by ``mtp_speculative.utils``.
    """

    def __init__(self, backend: ModeBackend, spec_mode: str, enable_dynmaic_mtp: bool) -> None:
        self.backend = backend
        self.proposer: BaseSpecProposer = build_spec_proposer(
            spec_mode=spec_mode,
            backend=backend,
            enable_dynmaic_mtp=enable_dynmaic_mtp,
        )
        self.planner: BaseMtpPlanner = self._build_mtp_planner(
            spec_mode=spec_mode,
            enable_dynmaic_mtp=enable_dynmaic_mtp,
        )

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

    # Decode planning.

    def plan_decode(self, model_input: ModelInput, decode_reqs: List) -> SpecDecodePlan:
        """Return the fixed or dynamic speculative plan for one decode iteration."""

        assert decode_reqs, "non-DP speculative decode requires at least one request"
        return self.planner.plan(
            decode_reqs=decode_reqs,
            origin_batch_size=model_input.batch_size,
        )

    def prepare_decode_model_input(
        self,
        model_input: ModelInput,
        req_num: int,
        plan: SpecDecodePlan,
    ) -> Tuple[ModelInput, Optional[AsyncPinnedCpuTensor]]:
        """Apply target verify-row compaction when the planned batch is smaller."""

        assert model_input.batch_size == plan.origin_batch_size
        if plan.dynamic_batch_size == plan.origin_batch_size:
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

    # Draft proposal generation.

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

    # Planner runtime statistics.

    def update_planner_statics(
        self,
        plan: SpecDecodePlan,
        proposal: SpecProposal,
        req_num: int,
        accept_lengths_cpu: torch.Tensor,
    ) -> None:
        """Update the current planner with iteration-level runtime statistics."""

        self.planner.update_statics(
            plan=plan,
            proposal=proposal,
            req_num=req_num,
            accept_lengths=accept_lengths_cpu,
        )

    def _build_mtp_planner(self, spec_mode: str, enable_dynmaic_mtp: bool) -> BaseMtpPlanner:
        if not enable_dynmaic_mtp:
            return FixedSpecPlanner(max_draft_step=self.backend.max_draft_step)
        if spec_mode == "dspark":
            return DSparkPlanner(backend=self.backend)
        return LightSpecPlanner(spec_mode=spec_mode, backend=self.backend)
