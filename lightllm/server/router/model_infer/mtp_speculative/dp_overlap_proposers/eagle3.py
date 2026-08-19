import torch

from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
from lightllm.server.router.model_infer.mtp_speculative.dp_overlap_proposers.base import BaseDpOverlapProposer
from lightllm.server.router.model_infer.mtp_speculative.dp_overlap_proposers.eagle_utils import (
    fill_dp_eagle_draft_model_kv_state_overlap,
    propose_next_dp_eagle_autoregressive_overlap,
)
from lightllm.server.router.model_infer.mtp_speculative.proposers.eagle_utils import (
    EagleSpecProposal,
    fill_eagle_draft_model_kv_state,
    propose_next_eagle,
)


class DpOverlapEagle3Proposer(BaseDpOverlapProposer):
    """DP ``eagle3`` proposer。"""

    def _map_draft_token_ids(self, draft_token_ids: torch.Tensor) -> torch.Tensor:
        return self.backend.draft_models[0].map_draft_vocab_to_main_vocab(draft_token_ids)

    def fill_draft_model_kv_state(
        self,
        target_model_input: ModelInput,
        target_model_output: ModelOutput,
        next_token_ids: torch.Tensor,
    ) -> None:
        fill_eagle_draft_model_kv_state(self, target_model_input, target_model_output, next_token_ids)

    def fill_draft_model_kv_state_overlap(
        self,
        target_model_input0: ModelInput,
        target_model_output0: ModelOutput,
        next_token_ids0: torch.Tensor,
        target_model_input1: ModelInput,
        target_model_output1: ModelOutput,
        next_token_ids1: torch.Tensor,
    ) -> None:
        fill_dp_eagle_draft_model_kv_state_overlap(
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
    ) -> EagleSpecProposal:
        return propose_next_eagle(
            self,
            main_model_input,
            main_model_output,
            next_token_ids,
            b_req_mtp_start_loc,
            draft_step,
            accept_len,
            self._map_draft_token_ids,
        )

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
    ) -> EagleSpecProposal:
        return propose_next_dp_eagle_autoregressive_overlap(
            self,
            main_model_input0,
            main_model_output0,
            next_token_ids0,
            real_verify_rows0,
            accept_len0,
            main_model_input1,
            main_model_output1,
            next_token_ids1,
            real_verify_rows1,
            accept_len1,
            draft_step,
            self._map_draft_token_ids,
        )
