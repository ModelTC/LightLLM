from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
from lightllm.server.router.model_infer.mtp_speculative.dp_overlap_planner import (
    BaseDpOverlapPlanner,
    build_dp_overlap_planner,
)
from lightllm.server.router.model_infer.mtp_speculative.dp_overlap_proposers import build_dp_overlap_spec_proposer
from lightllm.server.router.model_infer.mtp_speculative.dp_overlap_proposers.base import BaseDpOverlapProposer
from lightllm.server.router.model_infer.mtp_speculative.proposers.base import SpecProposal

if TYPE_CHECKING:
    from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend


class DPOverlapSpecEngine:
    """双 microbatch overlap draft 流程使用的 DP MTP engine。"""

    def __init__(self, backend: ModeBackend, spec_mode: str, enable_dynmaic_mtp: bool) -> None:
        self.proposer: BaseDpOverlapProposer = build_dp_overlap_spec_proposer(
            spec_mode=spec_mode,
            backend=backend,
            enable_dynmaic_mtp=enable_dynmaic_mtp,
        )
        self.planner: BaseDpOverlapPlanner = build_dp_overlap_planner(backend=backend)

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
        target_model_input0: ModelInput,  # batch_size = padded_verify_batch_size0
        target_model_output0: ModelOutput,  # logits: [padded_verify_batch_size0, vocab_size]
        target_next_token_ids0: torch.Tensor,  # [padded_verify_batch_size0]
        real_verify_rows0: int,
        accept_len0: Optional[torch.Tensor],  # [req_num0]
        target_model_input1: ModelInput,  # batch_size = padded_verify_batch_size1
        target_model_output1: ModelOutput,  # logits: [padded_verify_batch_size1, vocab_size]
        target_next_token_ids1: torch.Tensor,  # [padded_verify_batch_size1]
        real_verify_rows1: int,
        accept_len1: Optional[torch.Tensor],  # [req_num1]
    ) -> SpecProposal:
        return self.proposer.propose_next_overlap(
            target_model_input0=target_model_input0,
            target_model_output0=target_model_output0,
            target_next_token_ids0=target_next_token_ids0,
            real_verify_rows0=real_verify_rows0,
            accept_len0=accept_len0,
            target_model_input1=target_model_input1,
            target_model_output1=target_model_output1,
            target_next_token_ids1=target_next_token_ids1,
            real_verify_rows1=real_verify_rows1,
            accept_len1=accept_len1,
            draft_step=self.planner.get_draft_step(),
        )


__all__ = ["DPOverlapSpecEngine"]
