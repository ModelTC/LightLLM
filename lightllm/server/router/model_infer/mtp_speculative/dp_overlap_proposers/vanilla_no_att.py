import copy
from dataclasses import dataclass

import torch

from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
from lightllm.server.router.model_infer.mtp_speculative.dp_overlap_proposers.base import BaseDpOverlapProposer
from lightllm.server.router.model_infer.mtp_speculative.proposers.base import SpecProposal


@dataclass
class VanillaSpecProposal(SpecProposal):
    """DP-overlap Vanilla No-Att proposal with optional selected-token probabilities."""

    schedule_scores: torch.Tensor | None = None


class DpOverlapVanillaNoAttProposer(BaseDpOverlapProposer):
    """DP ``vanilla_no_att`` proposer。"""

    def fill_draft_model_kv_state(
        self,
        target_model_input: ModelInput,
        target_model_output: ModelOutput,
        target_next_token_ids: torch.Tensor,
    ) -> None:
        pass

    def fill_draft_model_kv_state_overlap(
        self,
        target_model_input0: ModelInput,
        target_model_output0: ModelOutput,
        target_next_token_ids0: torch.Tensor,
        target_model_input1: ModelInput,
        target_model_output1: ModelOutput,
        target_next_token_ids1: torch.Tensor,
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
    ) -> VanillaSpecProposal:
        assert accept_len0 is not None
        assert accept_len1 is not None

        verify_width = self.backend.max_draft_step + 1
        real_verify_rows = (int(real_verify_rows0), int(real_verify_rows1))
        req_num_by_batch = tuple(row_count // verify_width for row_count in real_verify_rows)
        req_start_rows = (
            torch.arange(0, real_verify_rows0, verify_width, device=target_next_token_ids0.device),
            torch.arange(0, real_verify_rows1, verify_width, device=target_next_token_ids1.device),
        )
        accepted_tail_rows = (
            req_start_rows[0] + accept_len0[: req_num_by_batch[0]] - 1,
            req_start_rows[1] + accept_len1[: req_num_by_batch[1]] - 1,
        )
        model_inputs = [copy.copy(target_model_input0), copy.copy(target_model_input1)]
        draft_token_ids = [target_next_token_ids0, target_next_token_ids1]
        draft_hiddens = [
            target_model_output0.mtp_collector.spec_hidden,
            target_model_output1.mtp_collector.spec_hidden,
        ]
        proposal_token_ids = target_next_token_ids0.new_empty((sum(req_num_by_batch), draft_step))
        req_offset = req_num_by_batch[0]

        for step in range(draft_step):
            for batch_index, model_input in enumerate(model_inputs):
                model_input.input_ids = draft_token_ids[batch_index]
                model_input.mtp_draft_input_hiddens = draft_hiddens[batch_index]

            draft_outputs = self.backend.draft_models[step].microbatch_overlap_decode(*model_inputs)
            for batch_index, draft_output in enumerate(draft_outputs):
                draft_hiddens[batch_index] = draft_output.mtp_collector.spec_hidden
                draft_token_ids[batch_index] = self.backend._gen_argmax_token_ids(draft_output)

            proposal_token_ids[:req_offset, step] = draft_token_ids[0].index_select(0, accepted_tail_rows[0].long())
            proposal_token_ids[req_offset:, step] = draft_token_ids[1].index_select(0, accepted_tail_rows[1].long())

        return VanillaSpecProposal(
            token_ids=proposal_token_ids,
            extra_mem_indexes_cpu=[],
            schedule_scores=None,
        )
