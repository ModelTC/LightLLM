import torch

from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
from lightllm.server.router.model_infer.mtp_speculative.dp_proposers.base import BaseDpProposer
from lightllm.server.router.model_infer.mtp_speculative.proposers.base import SpecProposal
from lightllm.server.router.model_infer.mtp_speculative.proposers.vanilla_utils import (
    build_chained_mtp_draft_state_from_prefill,
    propose_next_chained_mtp,
)


class DpVanillaNoAttProposer(BaseDpProposer):
    """普通 DP ``vanilla_no_att`` proposer。"""

    def build_draft_state_from_prefill(
        self,
        target_model_input: ModelInput,
        target_model_output: ModelOutput,
        next_token_ids: torch.Tensor,
    ) -> None:
        build_chained_mtp_draft_state_from_prefill(self, target_model_input, target_model_output, next_token_ids)

    def propose_next(
        self,
        main_model_input: ModelInput,
        main_model_output: ModelOutput,
        next_token_ids: torch.Tensor,
        b_req_mtp_start_loc: torch.Tensor,
        draft_step: int,
        accept_len: torch.Tensor | None = None,
    ) -> SpecProposal:
        return propose_next_chained_mtp(self, main_model_input, main_model_output, next_token_ids, draft_step)
