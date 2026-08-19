from typing import Optional

import torch

from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
from lightllm.server.router.model_infer.mtp_speculative.dp_planner import BaseDpPlanner, build_dp_planner
from lightllm.server.router.model_infer.mtp_speculative.dp_proposers import build_dp_spec_proposer
from lightllm.server.router.model_infer.mtp_speculative.dp_proposers.base import BaseDpProposer
from lightllm.server.router.model_infer.mtp_speculative.proposers.base import SpecProposal


class DPSpecEngine:
    """普通 DP prefill/decode 使用的 MTP engine。"""

    def __init__(self, backend, spec_mode: str, enable_dynmaic_mtp: bool) -> None:
        self.proposer: BaseDpProposer = build_dp_spec_proposer(
            spec_mode=spec_mode,
            backend=backend,
            enable_dynmaic_mtp=enable_dynmaic_mtp,
        )
        self.planner: BaseDpPlanner = build_dp_planner(backend=backend)

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

    def propose_next(
        self,
        main_model_input: ModelInput,
        main_model_output: ModelOutput,
        next_token_ids: torch.Tensor,
        b_req_mtp_start_loc: torch.Tensor,
        accept_len: Optional[torch.Tensor] = None,
    ) -> SpecProposal:
        return self.proposer.propose_next(
            main_model_input=main_model_input,
            main_model_output=main_model_output,
            next_token_ids=next_token_ids,
            b_req_mtp_start_loc=b_req_mtp_start_loc,
            draft_step=self.planner.get_draft_step(),
            accept_len=accept_len,
        )


__all__ = ["DPSpecEngine"]
