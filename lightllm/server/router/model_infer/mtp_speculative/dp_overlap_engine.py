from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
from lightllm.server.router.model_infer.mtp_speculative.dp_overlap_proposers import (
    build_dp_overlap_spec_proposer,
)
from lightllm.server.router.model_infer.mtp_speculative.dp_overlap_proposers.base import (
    BaseDpOverlapProposer,
)
from lightllm.server.router.model_infer.mtp_speculative.engine import SpecEngine
from lightllm.server.router.model_infer.mtp_speculative.planner import SpecDecodePlan
from lightllm.server.router.model_infer.mtp_speculative.proposers.base import (
    SpecProposal,
)
from lightllm.server.router.model_infer.pin_mem_manager import AsyncPinnedCpuTensor

if TYPE_CHECKING:
    from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend


class DPOverlapSpecEngine:
    """双 microbatch overlap draft 流程使用的 DP MTP engine。"""

    def __init__(
        self,
        backend: ModeBackend,
        spec_mode: str,
        enable_dynmaic_mtp: bool,
        common_engine: SpecEngine,
    ) -> None:
        self.common_engine = common_engine
        self.proposer: BaseDpOverlapProposer = build_dp_overlap_spec_proposer(
            spec_mode=spec_mode,
            backend=backend,
            enable_dynmaic_mtp=enable_dynmaic_mtp,
        )

    def plan_decode(
        self,
        model_input0: ModelInput,
        model_input1: ModelInput,
        decode_reqs: List,
    ) -> SpecDecodePlan:
        """Use the common planner for the combined two-microbatch layout."""

        return self.common_engine.planner.plan(
            decode_reqs=decode_reqs,
            origin_batch_size=model_input0.batch_size + model_input1.batch_size,
        )

    def prepare_decode_model_inputs(
        self,
        model_input0: ModelInput,
        req_num0: int,
        model_input1: ModelInput,
        req_num1: int,
        plan: SpecDecodePlan,
    ) -> Tuple[ModelInput, Optional[AsyncPinnedCpuTensor], ModelInput, Optional[AsyncPinnedCpuTensor],]:
        """Split the LightSpec verify budget and compact both microbatches.

        The combined budget is split in proportion to each side's real request
        count, so every request receives a similar average number of verify
        rows. Capacity overflow is transferred to the other side, and every
        request still keeps at least its target row.
        """

        origin_batch_size = model_input0.batch_size + model_input1.batch_size
        assert plan.origin_batch_size == origin_batch_size
        if plan.dynamic_batch_size == origin_batch_size:
            return model_input0, None, model_input1, None

        assert req_num0 == req_num1 or req_num0 == req_num1 + 1
        req_num = req_num0 + req_num1
        assert req_num > 0
        max_batch_size0 = min(model_input0.batch_size, req_num0 * (plan.pre_draft_step + 1))
        max_batch_size1 = min(model_input1.batch_size, req_num1 * (plan.pre_draft_step + 1))

        # 先计算每个请求平均可分到的 verify 行数，再按请求数分配给第一侧。
        avg_verify_rows_per_req = plan.dynamic_batch_size / req_num
        expected_batch_size0 = int(avg_verify_rows_per_req * req_num0 + 0.5)
        min_batch_size0 = max(req_num0, plan.dynamic_batch_size - max_batch_size1)
        max_batch_size0 = min(max_batch_size0, plan.dynamic_batch_size - req_num1)
        dynamic_batch_size0 = min(max(expected_batch_size0, min_batch_size0), max_batch_size0)
        dynamic_batch_size1 = plan.dynamic_batch_size - dynamic_batch_size0

        assert req_num0 <= dynamic_batch_size0 <= max_batch_size0
        assert req_num1 <= dynamic_batch_size1 <= max_batch_size1

        plan0 = SpecDecodePlan(
            origin_batch_size=model_input0.batch_size,
            dynamic_batch_size=dynamic_batch_size0,
            draft_step=plan.draft_step,
            pre_draft_step=plan.pre_draft_step,
            all_reqs_have_proposals=plan.all_reqs_have_proposals,
        )
        plan1 = SpecDecodePlan(
            origin_batch_size=model_input1.batch_size,
            dynamic_batch_size=dynamic_batch_size1,
            draft_step=plan.draft_step,
            pre_draft_step=plan.pre_draft_step,
            all_reqs_have_proposals=plan.all_reqs_have_proposals,
        )
        model_input0, selected_row_mask_cpu0 = self.common_engine.prepare_decode_model_input(
            model_input=model_input0,
            req_num=req_num0,
            plan=plan0,
        )
        model_input1, selected_row_mask_cpu1 = self.common_engine.prepare_decode_model_input(
            model_input=model_input1,
            req_num=req_num1,
            plan=plan1,
        )
        return (
            model_input0,
            selected_row_mask_cpu0,
            model_input1,
            selected_row_mask_cpu1,
        )

    def fill_draft_model_kv_state_overlap(
        self,
        target_model_input0: ModelInput,
        target_model_output0: ModelOutput,
        target_next_token_ids0: torch.Tensor,
        target_model_input1: ModelInput,
        target_model_output1: ModelOutput,
        target_next_token_ids1: torch.Tensor,
    ) -> None:
        self.proposer.fill_draft_model_kv_state_overlap(
            target_model_input0=target_model_input0,
            target_model_output0=target_model_output0,
            target_next_token_ids0=target_next_token_ids0,
            target_model_input1=target_model_input1,
            target_model_output1=target_model_output1,
            target_next_token_ids1=target_next_token_ids1,
        )

    def propose_next_overlap(
        self,
        target_model_input0: ModelInput,  # batch_size = verify_batch_size0
        target_model_output0: ModelOutput,  # logits: [verify_batch_size0, vocab_size]
        target_next_token_ids0: torch.Tensor,  # [verify_batch_size0]
        accept_len0: torch.Tensor,  # [real_req_num0]
        target_model_input1: ModelInput,  # batch_size = verify_batch_size1
        target_model_output1: ModelOutput,  # logits: [verify_batch_size1, vocab_size]
        target_next_token_ids1: torch.Tensor,  # [verify_batch_size1]
        accept_len1: torch.Tensor,  # [real_req_num1]
        draft_step: int,
    ) -> SpecProposal:
        assert target_next_token_ids0.shape == (target_model_input0.batch_size,)
        assert target_next_token_ids1.shape == (target_model_input1.batch_size,)
        assert accept_len0.ndim == 1
        assert accept_len1.ndim == 1

        return self.proposer.propose_next_overlap(
            target_model_input0=target_model_input0,
            target_model_output0=target_model_output0,
            target_next_token_ids0=target_next_token_ids0,
            accept_len0=accept_len0,
            target_model_input1=target_model_input1,
            target_model_output1=target_model_output1,
            target_next_token_ids1=target_next_token_ids1,
            accept_len1=accept_len1,
            draft_step=draft_step,
        )

    def update_planner_statics(
        self,
        plan: SpecDecodePlan,
        proposal: SpecProposal,
        req_num: int,
        accept_lengths_cpu: torch.Tensor,
    ) -> None:
        self.common_engine.update_planner_statics(
            plan=plan,
            proposal=proposal,
            req_num=req_num,
            accept_lengths_cpu=accept_lengths_cpu,
        )


__all__ = ["DPOverlapSpecEngine"]
