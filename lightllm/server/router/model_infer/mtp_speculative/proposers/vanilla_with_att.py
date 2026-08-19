import torch

from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
from lightllm.server.router.model_infer.mtp_speculative.proposers.base import BaseSpecProposer
from lightllm.server.router.model_infer.mtp_speculative.proposers.vanilla_utils import (
    VanillaSpecProposal,
    build_chained_mtp_draft_state_from_prefill,
    propose_next_chained_mtp,
)


class VanillaWithAttProposer(BaseSpecProposer):
    """使用 attention KV cache 的 Vanilla chained MTP proposer。"""

    def build_draft_state_from_prefill(
        self,
        target_model_input: ModelInput,
        target_model_output: ModelOutput,
        next_token_ids: torch.Tensor,
    ) -> None:
        build_chained_mtp_draft_state_from_prefill(
            proposer=self,
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
        draft_step: int,
        accept_len: torch.Tensor | None = None,
    ) -> VanillaSpecProposal:
        return propose_next_chained_mtp(
            proposer=self,
            main_model_input=main_model_input,
            main_model_output=main_model_output,
            next_token_ids=next_token_ids,
            b_req_mtp_start_loc=b_req_mtp_start_loc,
            draft_step=draft_step,
            accept_len=accept_len,
        )
