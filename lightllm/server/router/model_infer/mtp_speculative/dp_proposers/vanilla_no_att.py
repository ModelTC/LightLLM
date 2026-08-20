import copy
from dataclasses import dataclass

import torch

from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
from lightllm.server.router.model_infer.mtp_speculative.dp_proposers.base import BaseDpProposer
from lightllm.server.router.model_infer.mtp_speculative.proposers.base import SpecProposal


@dataclass
class VanillaSpecProposal(SpecProposal):
    """DP Vanilla No-Att proposal with optional selected-token probabilities."""

    schedule_scores: torch.Tensor | None = None


class DpVanillaNoAttProposer(BaseDpProposer):
    """普通 DP ``vanilla_no_att`` proposer。"""

    def fill_draft_model_kv_state(
        self,
        target_model_input: ModelInput,
        target_model_output: ModelOutput,
        target_next_token_ids: torch.Tensor,
    ) -> None:
        pass

    def propose_next(
        self,
        target_model_input: ModelInput,
        target_model_output: ModelOutput,
        target_next_token_ids: torch.Tensor,
        b_req_mtp_start_loc: torch.Tensor,
        draft_step: int,
        accept_len: torch.Tensor | None = None,
    ) -> VanillaSpecProposal:
        req_num = int(b_req_mtp_start_loc.shape[0])
        proposal_token_ids_by_step = []
        schedule_scores_by_step = []

        if draft_step == 0:
            return VanillaSpecProposal(
                token_ids=target_next_token_ids.new_empty((req_num, 0)),
                extra_mem_indexes_cpu=[],
                schedule_scores=(
                    torch.empty((req_num, 0), dtype=torch.float32, device=target_next_token_ids.device)
                    if self.enable_dynmaic_mtp
                    else None
                ),
            )

        assert accept_len is not None
        accepted_tail_rows = (b_req_mtp_start_loc + accept_len - 1).long()
        draft_token_ids = target_next_token_ids
        draft_hidden = target_model_output.mtp_collector.spec_hidden
        draft_input = copy.copy(target_model_input)

        for step in range(draft_step):
            draft_input.input_ids = draft_token_ids
            draft_input.mtp_draft_input_hiddens = draft_hidden
            draft_output = self.backend.draft_models[step].forward(draft_input)
            draft_hidden = draft_output.mtp_collector.spec_hidden

            if self.enable_dynmaic_mtp:
                draft_token_ids, draft_token_probs = self.backend._gen_argmax_token_ids_and_prob(draft_output)
                selected_token_probs = draft_token_probs.index_select(0, accepted_tail_rows)
                schedule_scores_by_step.append(selected_token_probs.float().unsqueeze(1))
            else:
                draft_token_ids = self.backend._gen_argmax_token_ids(draft_output)
            selected_token_ids = draft_token_ids.index_select(0, accepted_tail_rows)
            proposal_token_ids_by_step.append(selected_token_ids.unsqueeze(1))

        proposal_token_ids = torch.cat(proposal_token_ids_by_step, dim=1)
        schedule_scores = torch.cat(schedule_scores_by_step, dim=1) if self.enable_dynmaic_mtp else None
        return VanillaSpecProposal(
            token_ids=proposal_token_ids,
            extra_mem_indexes_cpu=[],
            schedule_scores=schedule_scores,
        )
