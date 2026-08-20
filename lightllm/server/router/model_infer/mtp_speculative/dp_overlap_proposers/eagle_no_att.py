import torch

from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
from lightllm.server.router.model_infer.mtp_speculative.dp_overlap_proposers.base import BaseDpOverlapProposer
from lightllm.server.router.model_infer.mtp_speculative.dp_overlap_proposers.eagle_utils import (
    fill_dp_eagle_draft_model_kv_state_overlap,
    fill_eagle_draft_model_kv_state,
    propose_next_dp_eagle_fixed_layout_overlap,
    propose_next_eagle,
)
from lightllm.server.router.model_infer.mtp_speculative.proposers.proposal_type import EagleSpecProposal


class DpOverlapEagleNoAttProposer(BaseDpOverlapProposer):
    """DP ``eagle_no_att`` proposer。"""

    def fill_draft_model_kv_state(
        self,
        target_model_input: ModelInput,
        target_model_output: ModelOutput,
        target_next_token_ids: torch.Tensor,
    ) -> None:
        fill_eagle_draft_model_kv_state(self, target_model_input, target_model_output, target_next_token_ids)

    def fill_draft_model_kv_state_overlap(
        self,
        target_model_input0: ModelInput,
        target_model_output0: ModelOutput,
        target_next_token_ids0: torch.Tensor,
        target_model_input1: ModelInput,
        target_model_output1: ModelOutput,
        target_next_token_ids1: torch.Tensor,
    ) -> None:
        fill_dp_eagle_draft_model_kv_state_overlap(
            self,
            target_model_input0,
            target_model_output0,
            target_next_token_ids0,
            target_model_input1,
            target_model_output1,
            target_next_token_ids1,
        )

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

    def propose_next_overlap(
        self,
        target_model_input0: ModelInput,
        target_model_output0: ModelOutput,
        target_next_token_ids0: torch.Tensor,
        real_verify_rows0: int,
        accept_len0: torch.Tensor | None,
        target_model_input1: ModelInput,
        target_model_output1: ModelOutput,
        target_next_token_ids1: torch.Tensor,
        real_verify_rows1: int,
        accept_len1: torch.Tensor | None,
        draft_step: int,
    ) -> EagleSpecProposal:
        return propose_next_dp_eagle_fixed_layout_overlap(
            self,
            target_model_input0,
            target_model_output0,
            target_next_token_ids0,
            real_verify_rows0,
            accept_len0,
            target_model_input1,
            target_model_output1,
            target_next_token_ids1,
            real_verify_rows1,
            accept_len1,
            draft_step,
            lambda token_ids: token_ids,
        )
