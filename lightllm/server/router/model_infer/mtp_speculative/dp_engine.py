from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
from lightllm.server.router.model_infer.mtp_speculative.dp_planner import BaseDpPlanner, build_dp_planner
from lightllm.server.router.model_infer.mtp_speculative.dp_proposers import build_dp_spec_proposer
from lightllm.server.router.model_infer.mtp_speculative.dp_proposers.base import BaseDpProposer
from lightllm.server.router.model_infer.mtp_speculative.proposers.base import SpecProposal

if TYPE_CHECKING:
    from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend


class DPSpecEngine:
    """普通 DP prefill/decode 使用的 MTP engine。"""

    def __init__(self, backend: ModeBackend, spec_mode: str, enable_dynmaic_mtp: bool) -> None:
        self.proposer: BaseDpProposer = build_dp_spec_proposer(
            spec_mode=spec_mode,
            backend=backend,
            enable_dynmaic_mtp=enable_dynmaic_mtp,
        )
        self.planner: BaseDpPlanner = build_dp_planner(backend=backend)

    def fill_draft_model_kv_state(
        self,
        target_model_input: ModelInput,
        target_model_output: ModelOutput,
        target_next_token_ids: torch.Tensor,
    ) -> None:
        self.proposer.fill_draft_model_kv_state(
            target_model_input=target_model_input,
            target_model_output=target_model_output,
            target_next_token_ids=target_next_token_ids,
        )

    def propose_next(
        self,
        target_model_input: ModelInput,  # batch_size = verify_batch_size
        target_model_output: ModelOutput,  # logits: [verify_batch_size, vocab_size]
        target_next_token_ids: torch.Tensor,  # [verify_batch_size]
        b_req_mtp_start_loc: torch.Tensor,  # [req_num]
        accept_len: Optional[torch.Tensor] = None,  # [req_num]
    ) -> SpecProposal:
        return self.proposer.propose_next(
            target_model_input=target_model_input,
            target_model_output=target_model_output,
            target_next_token_ids=target_next_token_ids,
            b_req_mtp_start_loc=b_req_mtp_start_loc,
            draft_step=self.planner.get_draft_step(),
            accept_len=accept_len,
        )


__all__ = ["DPSpecEngine"]
