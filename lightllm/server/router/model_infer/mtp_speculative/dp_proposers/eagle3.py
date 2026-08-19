import torch

from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
from lightllm.server.router.model_infer.mtp_speculative.dp_proposers.base import BaseDpProposer
from lightllm.server.router.model_infer.mtp_speculative.proposers.eagle_utils import (
    EagleSpecProposal,
    build_eagle_draft_state_from_prefill,
    propose_next_eagle,
)


class DpEagle3Proposer(BaseDpProposer):
    """普通 DP ``eagle3`` proposer。"""

    def _map_draft_token_ids(self, draft_token_ids: torch.Tensor) -> torch.Tensor:
        return self.backend.draft_models[0].map_draft_vocab_to_main_vocab(draft_token_ids)

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
            self._map_draft_token_ids,
        )
