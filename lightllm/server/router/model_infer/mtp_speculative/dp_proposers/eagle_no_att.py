import torch

from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
from lightllm.server.router.model_infer.mtp_speculative.dp_proposers.base import BaseDpProposer
from lightllm.server.router.model_infer.mtp_speculative.proposers.eagle_utils import (
    EagleSpecProposal,
    build_eagle_draft_state_from_prefill,
    propose_next_eagle,
)


class DpEagleNoAttProposer(BaseDpProposer):
    """普通 DP ``eagle_no_att`` proposer。"""

    def build_draft_state_from_prefill(
        self,
        target_model_input: ModelInput,
        target_model_output: ModelOutput,
        next_token_ids: torch.Tensor,
    ) -> None:
        build_eagle_draft_state_from_prefill(self, target_model_input, target_model_output, next_token_ids)

    def propose_next(
        self,
        main_model_input: ModelInput,
        main_model_output: ModelOutput,
        next_token_ids: torch.Tensor,
        b_req_mtp_start_loc: torch.Tensor,
        draft_step: int,
        accept_len: torch.Tensor | None = None,
    ) -> EagleSpecProposal:
        return propose_next_eagle(
            self,
            main_model_input,
            main_model_output,
            next_token_ids,
            b_req_mtp_start_loc,
            draft_step,
            accept_len,
            lambda token_ids: token_ids,
        )
