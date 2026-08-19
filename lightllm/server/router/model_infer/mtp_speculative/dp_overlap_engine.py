from typing import Optional

import torch

from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
from lightllm.server.router.model_infer.mtp_speculative.dp_overlap_planner import (
    BaseDpOverlapPlanner,
    build_dp_overlap_planner,
)
from lightllm.server.router.model_infer.mtp_speculative.dp_overlap_proposers import build_dp_overlap_spec_proposer
from lightllm.server.router.model_infer.mtp_speculative.dp_overlap_proposers.base import BaseDpOverlapProposer
from lightllm.server.router.model_infer.mtp_speculative.proposers.base import SpecProposal


class DPOverlapSpecEngine:
    """双 microbatch overlap draft 流程使用的 DP MTP engine。"""

    def __init__(self, backend, spec_mode: str, enable_dynmaic_mtp: bool) -> None:
        self.proposer: BaseDpOverlapProposer = build_dp_overlap_spec_proposer(
            spec_mode=spec_mode,
            backend=backend,
            enable_dynmaic_mtp=enable_dynmaic_mtp,
        )
        self.planner: BaseDpOverlapPlanner = build_dp_overlap_planner(backend=backend)

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

    def propose_next_overlap(
        self,
        main_model_input0: ModelInput,
        main_model_output0: ModelOutput,
        next_token_ids0: torch.Tensor,
        real_verify_rows0: int,
        accept_len0: Optional[torch.Tensor],
        main_model_input1: ModelInput,
        main_model_output1: ModelOutput,
        next_token_ids1: torch.Tensor,
        real_verify_rows1: int,
        accept_len1: Optional[torch.Tensor],
    ) -> SpecProposal:
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
            draft_step=self.planner.get_draft_step(),
        )


__all__ = ["DPOverlapSpecEngine"]
