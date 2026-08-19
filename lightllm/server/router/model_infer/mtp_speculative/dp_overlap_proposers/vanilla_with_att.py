import torch

from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
from lightllm.server.router.model_infer.mtp_speculative.dp_overlap_proposers.base import BaseDpOverlapProposer
from lightllm.server.router.model_infer.mtp_speculative.dp_overlap_proposers.vanilla_utils import (
    build_dp_chained_mtp_draft_state_from_prefill_overlap,
    propose_next_dp_chained_mtp_overlap,
)
from lightllm.server.router.model_infer.mtp_speculative.proposers.vanilla_utils import (
    VanillaSpecProposal,
    build_chained_mtp_draft_state_from_prefill,
    propose_next_chained_mtp,
)


class DpOverlapVanillaWithAttProposer(BaseDpOverlapProposer):
    """DP ``vanilla_with_att`` proposer。"""

    def build_draft_state_from_prefill(
        self,
        target_model_input: ModelInput,
        target_model_output: ModelOutput,
        next_token_ids: torch.Tensor,
    ) -> None:
        build_chained_mtp_draft_state_from_prefill(self, target_model_input, target_model_output, next_token_ids)

    def build_draft_state_from_prefill_overlap(
        self,
        target_model_input0: ModelInput,
        target_model_output0: ModelOutput,
        next_token_ids0: torch.Tensor,
        target_model_input1: ModelInput,
        target_model_output1: ModelOutput,
        next_token_ids1: torch.Tensor,
    ) -> None:
        build_dp_chained_mtp_draft_state_from_prefill_overlap(
            self,
            target_model_input0,
            target_model_output0,
            next_token_ids0,
            target_model_input1,
            target_model_output1,
            next_token_ids1,
        )

    def propose_next(
        self,
        main_model_input: ModelInput,
        main_model_output: ModelOutput,
        next_token_ids: torch.Tensor,
        b_req_mtp_start_loc: torch.Tensor,
        draft_step: int,
        accept_len: torch.Tensor | None = None,
    ) -> VanillaSpecProposal:
        return propose_next_chained_mtp(self, main_model_input, main_model_output, next_token_ids, draft_step)

    def propose_next_overlap(
        self,
        main_model_input0: ModelInput,
        main_model_output0: ModelOutput,
        next_token_ids0: torch.Tensor,
        real_verify_rows0: int,
        accept_len0: torch.Tensor | None,
        main_model_input1: ModelInput,
        main_model_output1: ModelOutput,
        next_token_ids1: torch.Tensor,
        real_verify_rows1: int,
        accept_len1: torch.Tensor | None,
        draft_step: int,
    ) -> VanillaSpecProposal:
        return propose_next_dp_chained_mtp_overlap(
            self,
            main_model_input0,
            main_model_output0,
            next_token_ids0,
            real_verify_rows0,
            main_model_input1,
            main_model_output1,
            next_token_ids1,
            real_verify_rows1,
            draft_step,
        )
