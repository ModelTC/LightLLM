import torch

from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
from lightllm.server.router.model_infer.mtp_speculative.proposers.base import BaseSpecProposer
from lightllm.server.router.model_infer.mtp_speculative.proposers.eagle_utils import (
    EagleSpecProposal,
    fill_eagle_draft_model_kv_state,
    generate_eagle_token_ids,
    generate_eagle_token_ids_and_prob,
    propose_next_eagle,
)


class Eagle3Proposer(BaseSpecProposer):
    """带有 draft-to-target vocabulary 映射的 EAGLE3 proposer。"""

    def _map_draft_token_ids(self, draft_token_ids: torch.Tensor) -> torch.Tensor:
        return self.backend.draft_models[0].map_draft_vocab_to_main_vocab(draft_token_ids)

    def _gen_argmax_token_ids(self, model_output: ModelOutput) -> torch.Tensor:
        return generate_eagle_token_ids(self, model_output, self._map_draft_token_ids)

    def _gen_argmax_token_ids_and_prob(self, model_output: ModelOutput):
        return generate_eagle_token_ids_and_prob(self, model_output, self._map_draft_token_ids)

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
            map_draft_token_ids=self._map_draft_token_ids,
        )
