from __future__ import annotations

import math

import torch

from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
from lightllm.server.router.model_infer.speculative.proposers.base import SpecProposal
from lightllm.server.router.model_infer.speculative.proposers.eagle_mtp import RecurrentEagleMTPProposer


class Eagle3Proposer(RecurrentEagleMTPProposer):
    """Eagle3 proposer.

    After target verification, Eagle3 commits the accepted target segment into
    the draft cache with target hidden states, then drafts the next proposal
    from that corrected state.
    """

    _DRAFT_PRUNE_SAFETY_FACTOR = 1.10
    _DRAFT_PRUNE_MIN_DEPTH = 4

    def get_draft_cost_ms(
        self,
        draft_infer_costs,
        req_num: int,
        verify_batch_size: int,
        draft_step: int,
    ) -> float:
        draft_cost_ms = draft_infer_costs.estimate(verify_batch_size)
        active_count = req_num
        draft_row_budget = max(1, verify_batch_size - req_num)
        for step in range(1, draft_step):
            active_count = self._get_pruned_active_count(
                current_count=active_count,
                draft_row_budget=draft_row_budget,
                next_depth=step + 1,
            )
            draft_cost_ms += draft_infer_costs.get(active_count)
        return draft_cost_ms

    def _get_pruned_active_count(
        self,
        current_count: int,
        draft_row_budget: int,
        next_depth: int,
    ) -> int:
        if next_depth < self._DRAFT_PRUNE_MIN_DEPTH or current_count <= 1:
            return current_count

        # A selected token at depth d consumes all d prefix draft rows from
        # that request.  Therefore at most L/d chains can reach depth d when
        # the next target verify has L draft-row slots.  Keep a small safety
        # margin, then retain the highest-survival chain frontiers.
        active_count = math.ceil(self._DRAFT_PRUNE_SAFETY_FACTOR * max(1, draft_row_budget) / next_depth)
        return min(current_count, max(1, active_count))

    def _map_draft_token_ids(self, draft_token_ids: torch.Tensor) -> torch.Tensor:
        return self.backend.draft_models[0].map_draft_vocab_to_main_vocab(draft_token_ids)

    def propose_next(
        self,
        main_model_input: ModelInput,
        main_model_output: ModelOutput,
        next_token_ids: torch.Tensor,
        b_req_mtp_start_loc: torch.Tensor,
        draft_step: int,
        accept_len: torch.Tensor | None = None,
    ) -> SpecProposal:
        verify_row_count = next_token_ids.shape[0]
        num_reqs = b_req_mtp_start_loc.shape[0]
        proposal_token_ids = next_token_ids.new_full(
            (verify_row_count, draft_step + 1),
            fill_value=1,
        )
        proposal_token_ids[:, 0].copy_(next_token_ids)
        collect_dynamic_probs = self.enable_dynamic_spec
        schedule_scores = (
            torch.zeros(
                (verify_row_count, draft_step),
                dtype=torch.float32,
                device=next_token_ids.device,
            )
            if collect_dynamic_probs
            else None
        )

        target_hidden = main_model_output.spec_hidden

        # Scatter consumes the accepted-tail row for each request; only those
        # rows need new draft columns after the commit step.
        selected_rows = self.select_accepted_tail_rows(
            b_req_mtp_start_loc=b_req_mtp_start_loc,
            accept_len=accept_len,
        )
        draft_model = self.backend.draft_models[0]
        draft_model_input = self.make_verify_extend_input(
            base_input=main_model_input,
            input_ids=next_token_ids,
            draft_hidden=target_hidden,
        )
        draft_model_output = draft_model.forward(draft_model_input)

        if draft_step == 0:
            return SpecProposal(
                token_ids=proposal_token_ids,
                extra_mem_indexes_cpu=None,
                schedule_scores=schedule_scores,
            )

        draft_logits = draft_model_output.logits.index_select(0, selected_rows)
        if collect_dynamic_probs:
            draft_next_token_ids, selected_draft_prob = self._gen_argmax_token_ids_and_prob(
                ModelOutput(logits=draft_logits)
            )
            schedule_scores[:, 0] = self.scatter_selected_step_probs(
                selected_rows=selected_rows,
                selected_probs=selected_draft_prob,
                verify_row_count=verify_row_count,
            )
            chain_survival = selected_draft_prob.float().clamp(0.01, 0.99)
        else:
            draft_next_token_ids = self._gen_argmax_token_ids(ModelOutput(logits=draft_logits))
            chain_survival = None
        draft_hidden = draft_model_output.spec_hidden.index_select(0, selected_rows)
        proposal_token_ids[selected_rows, 1] = draft_next_token_ids
        if draft_step == 1:
            return SpecProposal(
                token_ids=proposal_token_ids,
                extra_mem_indexes_cpu=None,
                schedule_scores=schedule_scores,
            )

        eagle_mem_indexes_cpu = self.alloc_extra_mem_indexes(num_reqs * (draft_step - 1))
        eagle_mem_indexes = eagle_mem_indexes_cpu.cuda(non_blocking=True)

        selected_seq_len = main_model_input.b_seq_len.index_select(0, selected_rows) + 1
        selected_req_idx = main_model_input.b_req_idx.index_select(0, selected_rows)
        selected_mtp_index = torch.zeros_like(selected_req_idx)
        selected_position_delta = (
            main_model_input.b_position_delta.index_select(0, selected_rows)
            if main_model_input.b_position_delta is not None
            else None
        )
        one_row_group_marks = torch.ones((num_reqs,), dtype=torch.int32, device=next_token_ids.device)
        draft_row_budget = max(1, int(main_model_input.batch_size) - int(num_reqs))

        for step in range(1, draft_step):
            next_depth = step + 1
            active_count = (
                self._get_pruned_active_count(
                    current_count=int(selected_rows.shape[0]),
                    draft_row_budget=draft_row_budget,
                    next_depth=next_depth,
                )
                if collect_dynamic_probs
                else int(selected_rows.shape[0])
            )
            if active_count < int(selected_rows.shape[0]):
                keep_rows = torch.topk(
                    chain_survival,
                    k=active_count,
                    largest=True,
                    sorted=False,
                ).indices
                selected_rows = selected_rows.index_select(0, keep_rows)
                draft_next_token_ids = draft_next_token_ids.index_select(0, keep_rows)
                draft_hidden = draft_hidden.index_select(0, keep_rows)
                selected_seq_len = selected_seq_len.index_select(0, keep_rows)
                selected_req_idx = selected_req_idx.index_select(0, keep_rows)
                selected_mtp_index = selected_mtp_index.index_select(0, keep_rows)
                if selected_position_delta is not None:
                    selected_position_delta = selected_position_delta.index_select(0, keep_rows)
                one_row_group_marks = one_row_group_marks.index_select(0, keep_rows)
                chain_survival = chain_survival.index_select(0, keep_rows)

            mem_start = (step - 1) * num_reqs
            mem_indexes_i = eagle_mem_indexes[mem_start : mem_start + active_count]
            draft_input = self.make_single_step_decode_input(
                base_input=main_model_input,
                input_ids=draft_next_token_ids,
                draft_hidden=draft_hidden,
                b_req_idx=selected_req_idx,
                b_mtp_index=selected_mtp_index,
                b_seq_len=selected_seq_len,
                b_position_delta=selected_position_delta,
                mem_indexes=mem_indexes_i,
                b_mark_shared_group=one_row_group_marks,
                max_kv_seq_len=main_model_input.max_kv_seq_len + step,
            )
            draft_output = draft_model.forward(draft_input)
            if collect_dynamic_probs:
                draft_next_token_ids, selected_draft_prob = self._gen_argmax_token_ids_and_prob(draft_output)
                schedule_scores[:, step] = self.scatter_selected_step_probs(
                    selected_rows=selected_rows,
                    selected_probs=selected_draft_prob,
                    verify_row_count=verify_row_count,
                )
                chain_survival = chain_survival * selected_draft_prob.float().clamp(0.01, 0.99)
            else:
                draft_next_token_ids = self._gen_argmax_token_ids(draft_output)
            proposal_token_ids[selected_rows, step + 1] = draft_next_token_ids
            draft_hidden = draft_output.spec_hidden
            selected_seq_len = selected_seq_len + 1

        return SpecProposal(
            token_ids=proposal_token_ids,
            extra_mem_indexes_cpu=eagle_mem_indexes_cpu,
            schedule_scores=schedule_scores,
        )
