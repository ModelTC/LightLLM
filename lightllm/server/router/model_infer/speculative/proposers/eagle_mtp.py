from __future__ import annotations

import copy

import torch

from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
from lightllm.server.router.model_infer.speculative.proposers.base import BaseSpecProposer, SpecProposal


class RecurrentEagleMTPProposer(BaseSpecProposer):
    """Shared draft-state setup for recurrent Eagle MTP proposers."""

    def get_draft_steps(self):
        return tuple(range(1, self.backend.max_draft_step + 1))

    def get_draft_cost_ms(
        self,
        draft_infer_costs,
        req_num: int,
        verify_batch_size: int,
        draft_step: int,
    ) -> float:
        # The mandatory extend processes all verified rows and produces the
        # first candidate. Later recurrent forwards process one row per request.
        extend_cost_ms = draft_infer_costs.estimate(verify_batch_size)
        decode_cost_ms = draft_infer_costs.get(req_num) * (draft_step - 1)
        return extend_cost_ms + decode_cost_ms

    def build_draft_state_from_prefill(
        self,
        target_model_input: ModelInput,
        target_model_output: ModelOutput,
        next_token_ids: torch.Tensor,
    ) -> None:
        from lightllm.server.router.model_infer.mode_backend.mtp_pre_process import prepare_mtp_prefill_inputs

        draft_model_input = prepare_mtp_prefill_inputs(
            model_input=target_model_input,
            b_next_token_ids=next_token_ids,
            mtp_draft_input_hiddens=target_model_output.spec_hidden,
        )
        self.backend.draft_models[0].forward(draft_model_input)

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

        draft_model_input0 = prepare_mtp_prefill_inputs(
            model_input=target_model_input0,
            b_next_token_ids=next_token_ids0,
            mtp_draft_input_hiddens=target_model_output0.spec_hidden,
        )
        draft_model_input1 = prepare_mtp_prefill_inputs(
            model_input=target_model_input1,
            b_next_token_ids=next_token_ids1,
            mtp_draft_input_hiddens=target_model_output1.spec_hidden,
        )
        self.backend.draft_models[0].microbatch_overlap_prefill(draft_model_input0, draft_model_input1)

    def _map_draft_token_ids(self, draft_token_ids: torch.Tensor) -> torch.Tensor:
        return draft_token_ids

    def _gen_argmax_token_ids(self, model_output: ModelOutput) -> torch.Tensor:
        return self._map_draft_token_ids(self.backend._gen_argmax_token_ids(model_output))

    def _gen_argmax_token_ids_and_prob(self, model_output: ModelOutput):
        draft_token_ids, draft_probs = self.backend._gen_argmax_token_ids_and_prob(model_output)
        return self._map_draft_token_ids(draft_token_ids), draft_probs

    def make_verify_extend_input(
        self,
        base_input: ModelInput,
        input_ids: torch.Tensor,
        draft_hidden: torch.Tensor,
    ) -> ModelInput:
        new_input = copy.copy(base_input)
        batch_size = int(input_ids.shape[0])
        new_input.is_prefill = True
        new_input.batch_size = batch_size
        new_input.total_token_num = batch_size
        new_input.prefix_total_token_num = 0
        new_input.max_q_seq_len = 1
        new_input.max_cache_len = max(0, int(base_input.max_cache_len or 0))
        new_input.input_ids = input_ids
        new_input.mtp_draft_input_hiddens = draft_hidden
        new_input.mem_indexes_cpu = None
        # Treat each verify row as a one-token extend request. This writes the
        # accepted prefix into draft KV without giving recurrent decode a
        # second attention topology or CUDA graph variant.
        new_input.b_ready_cache_len = base_input.b_seq_len - 1
        new_input.b_prefill_start_loc = torch.arange(
            batch_size,
            dtype=torch.int32,
            device=input_ids.device,
        )
        new_input.b_prefill_has_output_cpu = [False] * batch_size
        new_input.b_position_delta = None
        new_input.b_is_decode_req = None
        new_input.multimodal_params = [{"images": [], "audios": []}] * batch_size
        return new_input

    def make_single_step_decode_input(
        self,
        base_input: ModelInput,
        input_ids: torch.Tensor,
        draft_hidden: torch.Tensor,
        b_req_idx: torch.Tensor,
        b_mtp_index: torch.Tensor,
        b_seq_len: torch.Tensor,
        b_position_delta: torch.Tensor,
        mem_indexes: torch.Tensor,
        b_mark_shared_group: torch.Tensor,
        max_kv_seq_len: int,
    ) -> ModelInput:
        new_input = copy.copy(base_input)
        new_input.batch_size = b_seq_len.shape[0]
        new_input.input_ids = input_ids
        new_input.mtp_draft_input_hiddens = draft_hidden
        new_input.b_req_idx = b_req_idx
        new_input.b_mtp_index = b_mtp_index
        new_input.b_seq_len = b_seq_len
        new_input.b_position_delta = b_position_delta
        new_input.mem_indexes = mem_indexes
        new_input.mem_indexes_cpu = None
        new_input.b_mark_shared_group = b_mark_shared_group
        new_input.b_shared_seq_len = None
        new_input.max_q_seq_len = 1
        new_input.max_kv_seq_len = max_kv_seq_len
        new_input.total_token_num = new_input.batch_size * max_kv_seq_len
        new_input.draft_step = 0
        # Recurrent Eagle decode only needs a correctly sized placeholder
        # list.  Nested per-row allocations otherwise sit between graph
        # replays and extend the draft proposal critical path.
        empty_multimodal_params = {"images": [], "audios": []}
        new_input.multimodal_params = [empty_multimodal_params] * new_input.batch_size
        return new_input

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
        """Run recurrent Eagle with DP microbatch overlap.

        Target verification remains physically padded to ``B * (K + 1)`` for
        DP collectives.  Draft state is corrected once over those verify rows;
        recurrent draft decode then runs one row per logical request (plus
        HOLD rows required to keep the two microbatches shape-compatible).
        """

        verify_width = self.backend.max_draft_step + 1
        inputs = (main_model_input0, main_model_input1)
        outputs = (main_model_output0, main_model_output1)
        next_ids = (next_token_ids0, next_token_ids1)
        real_verify_rows = (int(real_verify_rows0), int(real_verify_rows1))
        accept_lens = (accept_len0, accept_len1)

        request_capacities = []
        real_request_nums = []
        selected_rows = []
        extend_inputs = []
        for model_input, model_output, token_ids, real_rows, accept_len in zip(
            inputs, outputs, next_ids, real_verify_rows, accept_lens
        ):
            request_capacity = model_input.batch_size // verify_width
            real_request_num = real_rows // verify_width
            starts = torch.arange(
                0,
                model_input.batch_size,
                verify_width,
                dtype=torch.int32,
                device=token_ids.device,
            )
            selected = self.select_accepted_tail_rows(
                b_req_mtp_start_loc=starts,
                accept_len=accept_len,
            )
            request_capacities.append(request_capacity)
            real_request_nums.append(real_request_num)
            selected_rows.append(selected)
            extend_inputs.append(
                self.make_verify_extend_input(
                    base_input=model_input,
                    input_ids=token_ids,
                    draft_hidden=model_output.spec_hidden,
                )
            )

        real_row_count = real_verify_rows0 + real_verify_rows1
        proposal_token_ids = next_token_ids0.new_full(
            (real_row_count, draft_step + 1),
            fill_value=1,
        )
        proposal_token_ids[:real_verify_rows0, 0].copy_(next_token_ids0[:real_verify_rows0])
        proposal_token_ids[real_verify_rows0:, 0].copy_(next_token_ids1[:real_verify_rows1])

        draft_model = self.backend.draft_models[0]
        extend_outputs = draft_model.microbatch_overlap_prefill(*extend_inputs)
        if draft_step == 0:
            return SpecProposal(
                token_ids=proposal_token_ids,
                extra_mem_indexes_cpu=None,
            )

        draft_next_token_ids = []
        draft_hiddens = []
        selected_seq_lens = []
        selected_req_idxs = []
        selected_mtp_idxs = []
        selected_position_deltas = []
        group_marks = []
        proposal_row_offsets = (0, real_verify_rows0)
        for index, (model_input, extend_output, selected, real_request_num) in enumerate(
            zip(inputs, extend_outputs, selected_rows, real_request_nums)
        ):
            selected_output = ModelOutput(logits=extend_output.logits.index_select(0, selected))
            step_token_ids = self._gen_argmax_token_ids(selected_output)
            draft_next_token_ids.append(step_token_ids)
            draft_hiddens.append(extend_output.spec_hidden.index_select(0, selected))
            selected_seq_lens.append(model_input.b_seq_len.index_select(0, selected) + 1)
            selected_req_idxs.append(model_input.b_req_idx.index_select(0, selected))
            selected_mtp_idxs.append(torch.zeros_like(selected_req_idxs[-1]))
            selected_position_deltas.append(
                model_input.b_position_delta.index_select(0, selected)
                if model_input.b_position_delta is not None
                else None
            )
            group_marks.append(
                torch.ones(
                    request_capacities[index],
                    dtype=torch.int32,
                    device=step_token_ids.device,
                )
            )
            if real_request_num > 0:
                proposal_rows = selected[:real_request_num].to(torch.long) + proposal_row_offsets[index]
                proposal_token_ids[proposal_rows, 1] = step_token_ids[:real_request_num]

        if draft_step == 1:
            return SpecProposal(
                token_ids=proposal_token_ids,
                extra_mem_indexes_cpu=None,
            )

        total_real_requests = sum(real_request_nums)
        extra_mem_indexes_cpu = self.alloc_extra_mem_indexes(total_real_requests * (draft_step - 1))
        extra_mem_indexes = extra_mem_indexes_cpu.cuda(non_blocking=True)
        hold_mem_index = self.backend.model.req_manager.mem_manager.HOLD_TOKEN_MEMINDEX

        for step in range(1, draft_step):
            mem_start = (step - 1) * total_real_requests
            step_mem_indexes = extra_mem_indexes[mem_start : mem_start + total_real_requests]
            step_inputs = []
            real_mem_start = 0
            for index, model_input in enumerate(inputs):
                real_request_num = real_request_nums[index]
                real_mem_indexes = step_mem_indexes[real_mem_start : real_mem_start + real_request_num]
                real_mem_start += real_request_num
                padded_mem_indexes = self._pad_step_mem_indexes(
                    real_mem_indexes=real_mem_indexes,
                    request_capacity=request_capacities[index],
                    hold_mem_index=hold_mem_index,
                )
                step_inputs.append(
                    self.make_single_step_decode_input(
                        base_input=model_input,
                        input_ids=draft_next_token_ids[index],
                        draft_hidden=draft_hiddens[index],
                        b_req_idx=selected_req_idxs[index],
                        b_mtp_index=selected_mtp_idxs[index],
                        b_seq_len=selected_seq_lens[index],
                        b_position_delta=selected_position_deltas[index],
                        mem_indexes=padded_mem_indexes,
                        b_mark_shared_group=group_marks[index],
                        max_kv_seq_len=model_input.max_kv_seq_len + step,
                    )
                )

            step_outputs = draft_model.microbatch_overlap_decode(*step_inputs)
            for index, step_output in enumerate(step_outputs):
                step_token_ids = self._gen_argmax_token_ids(step_output)
                draft_next_token_ids[index] = step_token_ids
                draft_hiddens[index] = step_output.spec_hidden
                selected_seq_lens[index] = selected_seq_lens[index] + 1
                real_request_num = real_request_nums[index]
                if real_request_num > 0:
                    proposal_rows = selected_rows[index][:real_request_num].to(torch.long) + proposal_row_offsets[index]
                    proposal_token_ids[proposal_rows, step + 1] = step_token_ids[:real_request_num]

        return SpecProposal(
            token_ids=proposal_token_ids,
            extra_mem_indexes_cpu=extra_mem_indexes_cpu,
        )


class EagleMTPProposer(RecurrentEagleMTPProposer):
    """Recurrent Eagle MTP proposer.

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
        compacted into a one-row recurrent batch on this path.
        """

        verify_width = self.backend.max_draft_step + 1
        model_inputs = (main_model_input0, main_model_input1)
        real_verify_rows = (int(real_verify_rows0), int(real_verify_rows1))
        real_request_nums = tuple(row_count // verify_width for row_count in real_verify_rows)
        request_capacities = tuple(model_input.batch_size // verify_width for model_input in model_inputs)
        total_real_requests = sum(real_request_nums)

        token_id_steps = [
            torch.cat(
                [
                    next_token_ids0[:real_verify_rows0],
                    next_token_ids1[:real_verify_rows1],
                ],
                dim=0,
            )
        ]
        if draft_step == 0:
            return SpecProposal(
                token_ids=token_id_steps[0].unsqueeze(1),
                extra_mem_indexes_cpu=None,
            )

        extra_mem_indexes_cpu = self.alloc_extra_mem_indexes(total_real_requests * draft_step)
        extra_mem_indexes = extra_mem_indexes_cpu.to(device=next_token_ids0.device, non_blocking=True)
        split = real_request_nums[0] * draft_step
        microbatch_mem_indexes = (
            extra_mem_indexes[:split],
            extra_mem_indexes[split:],
        )

        draft_token_ids = [next_token_ids0, next_token_ids1]
        draft_hiddens = [main_model_output0.spec_hidden, main_model_output1.spec_hidden]
        draft_model = self.backend.draft_models[0]
        hold_mem_index = self.backend.model.req_manager.mem_manager.HOLD_TOKEN_MEMINDEX

        for step in range(draft_step):
            for index, model_input in enumerate(model_inputs):
                model_input.input_ids = draft_token_ids[index]
                model_input.mtp_draft_input_hiddens = draft_hiddens[index]

            draft_outputs = draft_model.microbatch_overlap_decode(*model_inputs)

            for index, (model_input, draft_output) in enumerate(zip(model_inputs, draft_outputs)):
                model_input.b_seq_len += 1
                model_input.max_kv_seq_len += 1

                real_request_num = real_request_nums[index]
                mem_start = step * real_request_num
                step_mem_indexes = microbatch_mem_indexes[index][mem_start : mem_start + real_request_num]
                step_mem_indexes = self._pad_step_mem_indexes(
                    real_mem_indexes=step_mem_indexes,
                    request_capacity=request_capacities[index],
                    hold_mem_index=hold_mem_index,
                )
                model_input.mem_indexes = torch.cat(
                    [
                        model_input.mem_indexes.view(-1, verify_width)[:, 1:],
                        step_mem_indexes.view(-1, 1),
                    ],
                    dim=1,
                ).view(-1)

                draft_token_ids[index] = self._gen_argmax_token_ids(draft_output)
                draft_hiddens[index] = draft_output.spec_hidden

            token_id_steps.append(
                torch.cat(
                    [
                        draft_token_ids[0][:real_verify_rows0],
                        draft_token_ids[1][:real_verify_rows1],
                    ],
                    dim=0,
                )
            )

        return SpecProposal(
            token_ids=torch.stack(token_id_steps, dim=1),
            extra_mem_indexes_cpu=extra_mem_indexes_cpu,
        )

    def propose_next(
        self,
        main_model_input: ModelInput,
        main_model_output: ModelOutput,
        next_token_ids: torch.Tensor,
        b_req_mtp_start_loc: torch.Tensor,
        draft_step: int,
        accept_len: torch.Tensor | None = None,
    ) -> SpecProposal:
        verify_row_count = int(next_token_ids.shape[0])
        num_reqs = int(b_req_mtp_start_loc.shape[0])
        selected_rows = self.select_accepted_tail_rows(
            b_req_mtp_start_loc=b_req_mtp_start_loc,
            accept_len=accept_len,
        )
        proposal_token_ids = next_token_ids.new_full(
            (verify_row_count, draft_step + 1),
            fill_value=1,
        )
        proposal_token_ids[:, 0].copy_(next_token_ids)
        schedule_scores = (
            torch.zeros(
                (verify_row_count, draft_step),
                dtype=torch.float32,
                device=next_token_ids.device,
            )
            if self.enable_dynamic_spec
            else None
        )

        draft_model = self.backend.draft_models[0]
        extend_input = self.make_verify_extend_input(
            base_input=main_model_input,
            input_ids=next_token_ids,
            draft_hidden=main_model_output.spec_hidden,
        )
        extend_output = draft_model.forward(extend_input)

        if draft_step == 0:
            return SpecProposal(
                token_ids=proposal_token_ids,
                extra_mem_indexes_cpu=None,
                schedule_scores=schedule_scores,
            )

        selected_logits = extend_output.logits.index_select(0, selected_rows)
        selected_output = ModelOutput(logits=selected_logits)
        if self.enable_dynamic_spec:
            draft_next_token_ids, selected_prob = self._gen_argmax_token_ids_and_prob(selected_output)
            schedule_scores[:, 0] = self.scatter_selected_step_probs(
                selected_rows=selected_rows,
                selected_probs=selected_prob,
                verify_row_count=verify_row_count,
            )
        else:
            draft_next_token_ids = self._gen_argmax_token_ids(selected_output)
        proposal_token_ids[selected_rows, 1] = draft_next_token_ids
        draft_hidden = extend_output.spec_hidden.index_select(0, selected_rows)

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
        one_row_group_marks = torch.ones(num_reqs, dtype=torch.int32, device=next_token_ids.device)

        for step in range(1, draft_step):
            mem_start = (step - 1) * num_reqs
            draft_input = self.make_single_step_decode_input(
                base_input=main_model_input,
                input_ids=draft_next_token_ids,
                draft_hidden=draft_hidden,
                b_req_idx=selected_req_idx,
                b_mtp_index=selected_mtp_index,
                b_seq_len=selected_seq_len,
                b_position_delta=selected_position_delta,
                mem_indexes=eagle_mem_indexes[mem_start : mem_start + num_reqs],
                b_mark_shared_group=one_row_group_marks,
                max_kv_seq_len=main_model_input.max_kv_seq_len + step,
            )
            draft_output = draft_model.forward(draft_input)
            if self.enable_dynamic_spec:
                draft_next_token_ids, selected_prob = self._gen_argmax_token_ids_and_prob(draft_output)
                schedule_scores[:, step] = self.scatter_selected_step_probs(
                    selected_rows=selected_rows,
                    selected_probs=selected_prob,
                    verify_row_count=verify_row_count,
                )
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
