import torch

from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
from lightllm.server.router.model_infer.mtp_speculative.proposers.base import BaseSpecProposer
from lightllm.server.router.model_infer.mtp_speculative.proposers.eagle_utils import (
    EagleSpecProposal,
    fill_eagle_draft_model_kv_state,
    propose_next_eagle,
)


class EagleNoAttProposer(BaseSpecProposer):
    """不使用 attention KV cache 的 EAGLE proposer。"""

    def fill_draft_model_kv_state(
        self,
        target_model_input: ModelInput,
        target_model_output: ModelOutput,
        next_token_ids: torch.Tensor,
    ) -> None:
        fill_eagle_draft_model_kv_state(self, target_model_input, target_model_output, next_token_ids)

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
            proposer=self,
            main_model_input=main_model_input,
            main_model_output=main_model_output,
            next_token_ids=next_token_ids,
            b_req_mtp_start_loc=b_req_mtp_start_loc,
            draft_step=draft_step,
            accept_len=accept_len,
            map_draft_token_ids=lambda token_ids: token_ids,
        )
