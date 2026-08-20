import torch

from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
from lightllm.server.router.model_infer.mtp_speculative.dp_proposers.base import BaseDpProposer
from lightllm.server.router.model_infer.mtp_speculative.dp_proposers.eagle_utils import (
    fill_eagle_draft_model_kv_state,
    propose_next_eagle,
)
from lightllm.server.router.model_infer.mtp_speculative.proposers.proposal_type import EagleSpecProposal


class DpEagleWithAttProposer(BaseDpProposer):
    """普通 DP ``eagle_with_att`` proposer。"""

    def fill_draft_model_kv_state(
        self,
        target_model_input: ModelInput,
        target_model_output: ModelOutput,
        target_next_token_ids: torch.Tensor,
    ) -> None:
        fill_eagle_draft_model_kv_state(self, target_model_input, target_model_output, target_next_token_ids)

    def propose_next(
        self,
        target_model_input: ModelInput,
        target_model_output: ModelOutput,
        target_next_token_ids: torch.Tensor,
        b_req_mtp_start_loc: torch.Tensor,
        draft_step: int,
        accept_len: torch.Tensor | None = None,
    ) -> EagleSpecProposal:
        return propose_next_eagle(
            self,
            target_model_input,
            target_model_output,
            target_next_token_ids,
            b_req_mtp_start_loc,
            draft_step,
            accept_len,
            lambda token_ids: token_ids,
        )
