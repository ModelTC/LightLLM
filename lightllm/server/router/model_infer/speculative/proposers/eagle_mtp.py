from __future__ import annotations

import copy

import torch

from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
from lightllm.server.router.model_infer.speculative.proposers.base import BaseSpecProposer, SpecProposal


class AutoregressiveEagleProposer(BaseSpecProposer):
    """Shared autoregressive drafting flow for EAGLE-family proposers."""

    def get_draft_steps(self):
        return tuple(range(1, self.backend.max_draft_step + 1))

    def get_draft_cost_ms(
        self,
        draft_infer_costs,
        req_num: int,
        verify_batch_size: int,
        draft_step: int,
    ) -> float:
        draft_cost_ms = draft_infer_costs.estimate(verify_batch_size)
        if draft_step > 1:
            draft_cost_ms += draft_infer_costs.get(req_num) * (draft_step - 1)
        return draft_cost_ms

    def build_draft_state_from_prefill(
        self,
        target_model_input: ModelInput,
        target_model_output: ModelOutput,
        next_token_ids: torch.Tensor,
    ) -> None:
        from lightllm.server.router.model_infer.mode_backend.mtp_pre_process import prepare_mtp_prefill_inputs

        prepare_mtp_prefill_inputs(
            model_input=target_model_input,
            b_next_token_ids=next_token_ids,
            mtp_draft_input_hiddens=target_model_output.spec_hidden,
        )
        self.backend.draft_models[0].forward(target_model_input)

    def build_draft_state_from_prefill_overlap(
        self,
        target_model_input0: ModelInput,
        target_model_output0: ModelOutput,
        next_token_ids0: torch.Tensor,
        target_model_input1: ModelInput,
        target_model_output1: ModelOutput,
        next_token_ids1: torch.Tensor,
    ) -> None:
        from lightllm.server.router.model_infer.mode_backend.mtp_pre_process import prepare_mtp_prefill_inputs

        prepare_mtp_prefill_inputs(
            model_input=target_model_input0,
            b_next_token_ids=next_token_ids0,
            mtp_draft_input_hiddens=target_model_output0.spec_hidden,
        )
        prepare_mtp_prefill_inputs(
            model_input=target_model_input1,
            b_next_token_ids=next_token_ids1,
            mtp_draft_input_hiddens=target_model_output1.spec_hidden,
        )
        self.backend.draft_models[0].microbatch_overlap_prefill(target_model_input0, target_model_input1)

    def _map_draft_token_ids(self, draft_token_ids: torch.Tensor) -> torch.Tensor:
        return draft_token_ids

    def _gen_argmax_token_ids(self, model_output: ModelOutput) -> torch.Tensor:
        return self._map_draft_token_ids(self.backend._gen_argmax_token_ids(model_output))

    def _gen_argmax_token_ids_and_prob(self, model_output: ModelOutput):
        draft_token_ids, draft_token_probs = self.backend._gen_argmax_token_ids_and_prob(model_output)
        return self._map_draft_token_ids(draft_token_ids), draft_token_probs

    def prepare_verify_extend_input(
        self,
        model_input: ModelInput,
        input_ids: torch.Tensor,
        target_hidden: torch.Tensor,
    ) -> None:
        model_input.is_prefill = True
        model_input.total_token_num = model_input.batch_size
        model_input.prefix_total_token_num = 0
        model_input.max_cache_len = max(0, int(model_input.max_cache_len or 0))
        model_input.input_ids = input_ids
        model_input.mtp_draft_input_hiddens = target_hidden
        model_input.b_ready_cache_len = model_input.b_seq_len - 1
        model_input.b_prefill_start_loc = torch.arange(
            model_input.batch_size,
            dtype=torch.int32,
            device=input_ids.device,
        )

    @staticmethod
    def _pad_step_mem_indexes(
        real_mem_indexes: torch.Tensor,
        request_capacity: int,
        hold_mem_index: int,
    ) -> torch.Tensor:
        padded = torch.full(
            (request_capacity,),
            hold_mem_index,
            dtype=real_mem_indexes.dtype,
            device=real_mem_indexes.device,
        )
        padded[: real_mem_indexes.shape[0]].copy_(real_mem_indexes)
        return padded

    def propose_next(
        self,
        main_model_input: ModelInput,
        main_model_output: ModelOutput,
        next_token_ids: torch.Tensor,
        b_req_mtp_start_loc: torch.Tensor,
        draft_step: int,
        accept_len: torch.Tensor | None = None,
    ) -> SpecProposal:
        """Run the common autoregressive EAGLE proposal flow."""

        verify_row_count = int(next_token_ids.shape[0])
        request_count = int(b_req_mtp_start_loc.shape[0])
        accepted_tail_rows = (b_req_mtp_start_loc + accept_len - 1).long()
        proposal_token_ids = next_token_ids.new_full(
            (verify_row_count, draft_step + 1),
            fill_value=1,
        )
        proposal_token_ids[:, 0].copy_(next_token_ids)
        collect_schedule_scores = self.enable_dynmaic_mtp
        schedule_scores = (
            torch.zeros(
                (verify_row_count, draft_step),
                dtype=torch.float32,
                device=next_token_ids.device,
            )
            if collect_schedule_scores
            else None
        )

        draft_model = self.backend.draft_models[0]
        position_delta = main_model_input.b_position_delta
        self.prepare_verify_extend_input(
            model_input=main_model_input,
            input_ids=next_token_ids,
            target_hidden=main_model_output.spec_hidden,
        )
        extend_output = draft_model.forward(main_model_input)

        accepted_tail_output = ModelOutput(logits=extend_output.logits.index_select(0, accepted_tail_rows))
        if collect_schedule_scores:
            draft_token_ids, draft_token_probs = self._gen_argmax_token_ids_and_prob(accepted_tail_output)
            schedule_scores[accepted_tail_rows, 0] = draft_token_probs.float()
        else:
            draft_token_ids = self._gen_argmax_token_ids(accepted_tail_output)
        proposal_token_ids[accepted_tail_rows, 1] = draft_token_ids
        draft_hidden = extend_output.spec_hidden.index_select(0, accepted_tail_rows)

        if draft_step == 1:
            return SpecProposal(
                token_ids=proposal_token_ids,
                extra_mem_indexes_cpu=None,
                schedule_scores=schedule_scores,
            )

        extra_mem_indexes_cpu = self.alloc_extra_mem_indexes(request_count * (draft_step - 1))
        extra_mem_indexes = extra_mem_indexes_cpu.to(device=next_token_ids.device, non_blocking=True)
        draft_seq_lens = main_model_input.b_seq_len.index_select(0, accepted_tail_rows) + 1
        max_kv_seq_len = main_model_input.max_kv_seq_len
        draft_input = copy.copy(main_model_input)
        draft_input.is_prefill = False
        draft_input.batch_size = request_count
        draft_input.b_req_idx = main_model_input.b_req_idx.index_select(0, accepted_tail_rows)
        draft_input.b_mtp_index = torch.zeros_like(draft_input.b_req_idx)
        draft_input.b_seq_len = draft_seq_lens
        draft_input.b_position_delta = (
            position_delta.index_select(0, accepted_tail_rows) if position_delta is not None else None
        )
        draft_input.b_mark_shared_group = torch.ones_like(draft_input.b_req_idx)
        draft_input.b_shared_seq_len = None
        if len(draft_input.multimodal_params) != request_count:
            empty_multimodal_params = {"images": [], "audios": []}
            draft_input.multimodal_params = [empty_multimodal_params] * request_count

        for step in range(1, draft_step):
            mem_start = (step - 1) * request_count
            draft_input.input_ids = draft_token_ids
            draft_input.mtp_draft_input_hiddens = draft_hidden
            draft_input.mem_indexes = extra_mem_indexes[mem_start : mem_start + request_count]
            draft_input.max_kv_seq_len = max_kv_seq_len + step
            draft_input.total_token_num = request_count * draft_input.max_kv_seq_len
            draft_output = draft_model.forward(draft_input)
            if collect_schedule_scores:
                draft_token_ids, draft_token_probs = self._gen_argmax_token_ids_and_prob(draft_output)
                schedule_scores[accepted_tail_rows, step] = draft_token_probs.float()
            else:
                draft_token_ids = self._gen_argmax_token_ids(draft_output)
            proposal_token_ids[accepted_tail_rows, step + 1] = draft_token_ids
            draft_hidden = draft_output.spec_hidden
            draft_seq_lens.add_(1)

        return SpecProposal(
            token_ids=proposal_token_ids,
            extra_mem_indexes_cpu=extra_mem_indexes_cpu,
            schedule_scores=schedule_scores,
        )

    def propose_next_overlap(
        self,
        main_model_input0: ModelInput,
        main_model_output0: ModelOutput,
        next_token_ids0: torch.Tensor,
        real_verify_rows0: int,
        accept_len0: torch.Tensor,
        main_model_input1: ModelInput,
        main_model_output1: ModelOutput,
        next_token_ids1: torch.Tensor,
        real_verify_rows1: int,
        accept_len1: torch.Tensor,
        draft_step: int,
    ) -> SpecProposal:
        """Run autoregressive EAGLE drafting with DP microbatch overlap.

        Target verification remains physically padded to ``B * (K + 1)`` for
        DP collectives.  Draft state is corrected once over those verify rows;
        autoregressive drafting then runs one row per logical request (plus
        HOLD rows required to keep the two microbatches shape-compatible).
        """

        verify_width = self.backend.max_draft_step + 1
        model_inputs = (main_model_input0, main_model_input1)
        model_outputs = (main_model_output0, main_model_output1)
        next_token_ids_by_batch = (next_token_ids0, next_token_ids1)
        real_verify_row_counts = (int(real_verify_rows0), int(real_verify_rows1))
        accept_lens_by_batch = (accept_len0, accept_len1)
        position_deltas_by_batch = tuple(model_input.b_position_delta for model_input in model_inputs)
        max_kv_seq_lens_by_batch = tuple(model_input.max_kv_seq_len for model_input in model_inputs)

        request_capacities_by_batch = []
        real_request_counts = []
        accepted_tail_rows_by_batch = []
        for model_input, model_output, token_ids, real_verify_row_count, accept_len in zip(
            model_inputs,
            model_outputs,
            next_token_ids_by_batch,
            real_verify_row_counts,
            accept_lens_by_batch,
        ):
            request_capacity = model_input.batch_size // verify_width
            real_request_count = real_verify_row_count // verify_width
            starts = torch.arange(
                0,
                model_input.batch_size,
                verify_width,
                dtype=torch.int32,
                device=token_ids.device,
            )
            accepted_tail_rows = (starts + accept_len - 1).long()
            request_capacities_by_batch.append(request_capacity)
            real_request_counts.append(real_request_count)
            accepted_tail_rows_by_batch.append(accepted_tail_rows)
            self.prepare_verify_extend_input(
                model_input=model_input,
                input_ids=token_ids,
                target_hidden=model_output.spec_hidden,
            )

        verify_row_count = real_verify_rows0 + real_verify_rows1
        proposal_token_ids = next_token_ids0.new_full(
            (verify_row_count, draft_step + 1),
            fill_value=1,
        )
        proposal_token_ids[:real_verify_rows0, 0].copy_(next_token_ids0[:real_verify_rows0])
        proposal_token_ids[real_verify_rows0:, 0].copy_(next_token_ids1[:real_verify_rows1])

        draft_model = self.backend.draft_models[0]
        extend_outputs = draft_model.microbatch_overlap_prefill(*model_inputs)

        draft_token_ids_by_batch = []
        draft_hiddens_by_batch = []
        draft_seq_lens_by_batch = []
        draft_req_indices_by_batch = []
        proposal_rows_by_batch = []
        proposal_row_offsets = (0, real_verify_rows0)
        for batch_index, (model_input, extend_output, accepted_tail_rows, real_request_count) in enumerate(
            zip(model_inputs, extend_outputs, accepted_tail_rows_by_batch, real_request_counts)
        ):
            accepted_tail_output = ModelOutput(logits=extend_output.logits.index_select(0, accepted_tail_rows))
            draft_token_ids = self._gen_argmax_token_ids(accepted_tail_output)
            draft_token_ids_by_batch.append(draft_token_ids)
            draft_hiddens_by_batch.append(extend_output.spec_hidden.index_select(0, accepted_tail_rows))
            draft_seq_lens_by_batch.append(model_input.b_seq_len.index_select(0, accepted_tail_rows) + 1)
            draft_req_indices_by_batch.append(model_input.b_req_idx.index_select(0, accepted_tail_rows))
            proposal_rows = accepted_tail_rows[:real_request_count] + proposal_row_offsets[batch_index]
            proposal_rows_by_batch.append(proposal_rows)
            proposal_token_ids[proposal_rows, 1] = draft_token_ids[:real_request_count]

        if draft_step == 1:
            return SpecProposal(
                token_ids=proposal_token_ids,
                extra_mem_indexes_cpu=None,
            )

        for batch_index, model_input in enumerate(model_inputs):
            model_input.is_prefill = False
            model_input.batch_size = request_capacities_by_batch[batch_index]
            model_input.b_req_idx = draft_req_indices_by_batch[batch_index]
            model_input.b_mtp_index = torch.zeros_like(model_input.b_req_idx)
            model_input.b_seq_len = draft_seq_lens_by_batch[batch_index]
            model_input.b_position_delta = (
                position_deltas_by_batch[batch_index].index_select(0, accepted_tail_rows_by_batch[batch_index])
                if position_deltas_by_batch[batch_index] is not None
                else None
            )
            model_input.b_mark_shared_group = torch.ones_like(model_input.b_req_idx)
            model_input.b_shared_seq_len = None
            if len(model_input.multimodal_params) != model_input.batch_size:
                empty_multimodal_params = {"images": [], "audios": []}
                model_input.multimodal_params = [empty_multimodal_params] * model_input.batch_size

        total_real_request_count = sum(real_request_counts)
        extra_mem_indexes_cpu = self.alloc_extra_mem_indexes(total_real_request_count * (draft_step - 1))
        extra_mem_indexes = extra_mem_indexes_cpu.to(device=next_token_ids0.device, non_blocking=True)
        hold_mem_index = self.backend.model.req_manager.mem_manager.HOLD_TOKEN_MEMINDEX

        for step in range(1, draft_step):
            mem_start = (step - 1) * total_real_request_count
            step_mem_indexes = extra_mem_indexes[mem_start : mem_start + total_real_request_count]
            real_mem_start = 0
            for batch_index, model_input in enumerate(model_inputs):
                real_request_count = real_request_counts[batch_index]
                real_mem_indexes = step_mem_indexes[real_mem_start : real_mem_start + real_request_count]
                real_mem_start += real_request_count
                padded_mem_indexes = self._pad_step_mem_indexes(
                    real_mem_indexes=real_mem_indexes,
                    request_capacity=request_capacities_by_batch[batch_index],
                    hold_mem_index=hold_mem_index,
                )
                model_input.input_ids = draft_token_ids_by_batch[batch_index]
                model_input.mtp_draft_input_hiddens = draft_hiddens_by_batch[batch_index]
                model_input.mem_indexes = padded_mem_indexes
                model_input.max_kv_seq_len = max_kv_seq_lens_by_batch[batch_index] + step
                model_input.total_token_num = model_input.batch_size * model_input.max_kv_seq_len

            draft_outputs = draft_model.microbatch_overlap_decode(*model_inputs)
            for batch_index, draft_output in enumerate(draft_outputs):
                draft_token_ids = self._gen_argmax_token_ids(draft_output)
                draft_token_ids_by_batch[batch_index] = draft_token_ids
                draft_hiddens_by_batch[batch_index] = draft_output.spec_hidden
                draft_seq_lens_by_batch[batch_index].add_(1)
                real_request_count = real_request_counts[batch_index]
                proposal_token_ids[proposal_rows_by_batch[batch_index], step + 1] = draft_token_ids[:real_request_count]

        return SpecProposal(
            token_ids=proposal_token_ids,
            extra_mem_indexes_cpu=extra_mem_indexes_cpu,
        )


class EagleMTPProposer(AutoregressiveEagleProposer):
    """Autoregressive EAGLE proposer backed by an MTP draft model.

    The draft model keeps a cache and repeatedly feeds the previous proposal
    hidden back into the same draft model.
    """

    def propose_next_overlap(
        self,
        main_model_input0: ModelInput,
        main_model_output0: ModelOutput,
        next_token_ids0: torch.Tensor,
        real_verify_rows0: int,
        accept_len0: torch.Tensor,
        main_model_input1: ModelInput,
        main_model_output1: ModelOutput,
        next_token_ids1: torch.Tensor,
        real_verify_rows1: int,
        accept_len1: torch.Tensor,
        draft_step: int,
    ) -> SpecProposal:
        """Run the fixed-layout Eagle MTP decode for two DP microbatches.

        DPEP overlap requires every rank to keep the target verify layout
        throughout draft decode.  Each logical request therefore remains a
        contiguous group of ``draft_step + 1`` rows; accepted rows are not
        compacted into a one-row autoregressive draft batch on this path.
        """

        verify_width = self.backend.max_draft_step + 1
        model_inputs = (main_model_input0, main_model_input1)
        real_verify_row_counts = (int(real_verify_rows0), int(real_verify_rows1))
        real_request_counts = tuple(row_count // verify_width for row_count in real_verify_row_counts)
        request_capacities_by_batch = tuple(model_input.batch_size // verify_width for model_input in model_inputs)
        total_real_request_count = sum(real_request_counts)

        proposal_token_ids = next_token_ids0.new_empty((sum(real_verify_row_counts), draft_step + 1))
        proposal_token_ids[:real_verify_rows0, 0] = next_token_ids0[:real_verify_rows0]
        proposal_token_ids[real_verify_rows0:, 0] = next_token_ids1[:real_verify_rows1]

        extra_mem_indexes_cpu = self.alloc_extra_mem_indexes(total_real_request_count * draft_step)
        extra_mem_indexes = extra_mem_indexes_cpu.to(device=next_token_ids0.device, non_blocking=True)
        split = real_request_counts[0] * draft_step
        extra_mem_indexes_by_batch = (
            extra_mem_indexes[:split],
            extra_mem_indexes[split:],
        )

        draft_token_ids_by_batch = [next_token_ids0, next_token_ids1]
        draft_hiddens_by_batch = [main_model_output0.spec_hidden, main_model_output1.spec_hidden]
        draft_model = self.backend.draft_models[0]
        hold_mem_index = self.backend.model.req_manager.mem_manager.HOLD_TOKEN_MEMINDEX

        for step in range(draft_step):
            for batch_index, model_input in enumerate(model_inputs):
                model_input.input_ids = draft_token_ids_by_batch[batch_index]
                model_input.mtp_draft_input_hiddens = draft_hiddens_by_batch[batch_index]

            draft_outputs = draft_model.microbatch_overlap_decode(*model_inputs)

            for batch_index, (model_input, draft_output) in enumerate(zip(model_inputs, draft_outputs)):
                model_input.b_seq_len += 1
                model_input.max_kv_seq_len += 1

                real_request_count = real_request_counts[batch_index]
                mem_start = step * real_request_count
                step_mem_indexes = extra_mem_indexes_by_batch[batch_index][mem_start : mem_start + real_request_count]
                step_mem_indexes = self._pad_step_mem_indexes(
                    real_mem_indexes=step_mem_indexes,
                    request_capacity=request_capacities_by_batch[batch_index],
                    hold_mem_index=hold_mem_index,
                )
                model_input.mem_indexes = torch.cat(
                    [
                        model_input.mem_indexes.view(-1, verify_width)[:, 1:],
                        step_mem_indexes.view(-1, 1),
                    ],
                    dim=1,
                ).view(-1)

                draft_token_ids_by_batch[batch_index] = self._gen_argmax_token_ids(draft_output)
                draft_hiddens_by_batch[batch_index] = draft_output.spec_hidden

            proposal_token_ids[:real_verify_rows0, step + 1] = draft_token_ids_by_batch[0][:real_verify_rows0]
            proposal_token_ids[real_verify_rows0:, step + 1] = draft_token_ids_by_batch[1][:real_verify_rows1]

        return SpecProposal(
            token_ids=proposal_token_ids,
            extra_mem_indexes_cpu=extra_mem_indexes_cpu,
        )
