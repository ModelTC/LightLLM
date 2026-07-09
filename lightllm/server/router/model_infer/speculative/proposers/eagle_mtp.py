from __future__ import annotations

import copy

import torch

from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
from lightllm.server.router.model_infer.speculative.proposers.base import SpecProposal
from lightllm.server.router.model_infer.speculative.proposers.vanilla_mtp import VanillaMTPProposer


class RecurrentEagleMTPProposer(VanillaMTPProposer):
    """Shared draft-state setup for recurrent Eagle MTP proposers."""

    def build_initial_draft_state(
        self,
        *,
        model_input: ModelInput,
        next_token_ids: torch.Tensor,
    ) -> None:
        draft_model_input = self.prepare_draft_prefill_input(
            model_input=model_input,
            next_token_ids=next_token_ids,
        )
        self.backend.draft_models[0].forward(draft_model_input)
        return None

    def build_initial_draft_state_overlap(
        self,
        *,
        model_input0: ModelInput,
        next_token_ids0: torch.Tensor,
        model_input1: ModelInput,
        next_token_ids1: torch.Tensor,
    ) -> None:
        draft_model_input0 = self.prepare_draft_prefill_input(
            model_input=model_input0,
            next_token_ids=next_token_ids0,
            microbatch_index=0,
        )
        draft_model_input1 = self.prepare_draft_prefill_input(
            model_input=model_input1,
            next_token_ids=next_token_ids1,
            microbatch_index=1,
        )
        self.backend.draft_models[0].microbatch_overlap_prefill(draft_model_input0, draft_model_input1)
        return None

    def project_draft_decode_hidden(self, draft_hidden: torch.Tensor) -> torch.Tensor:
        if draft_hidden is None:
            return None

        draft_model = self.backend.draft_models[0]
        pre_infer = getattr(draft_model, "pre_infer", None)
        projector = getattr(pre_infer, "project_mtp_draft_hiddens", None)
        if projector is None:
            return draft_hidden

        # Keep draft decode CUDA graph input shape stable across target and draft hidden sources.
        return projector(
            draft_hidden,
            draft_model.pre_post_weight,
            use_custom_tensor_mananger=False,
        )

    def make_full_verify_decode_input(
        self,
        *,
        base_input: ModelInput,
        input_ids: torch.Tensor,
        draft_hidden: torch.Tensor,
    ) -> ModelInput:
        new_input = copy.copy(base_input)
        new_input.input_ids = input_ids
        new_input.mtp_draft_input_hiddens = draft_hidden
        new_input.mem_indexes_cpu = None
        new_input.disable_mtp_decode_att = False
        return new_input

    def make_single_step_decode_input(
        self,
        *,
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
        new_input.disable_mtp_decode_att = True
        new_input.max_q_seq_len = 1
        new_input.max_kv_seq_len = max_kv_seq_len
        new_input.total_token_num = new_input.batch_size * max_kv_seq_len
        new_input.multimodal_params = [{"images": [], "audios": []} for _ in range(new_input.batch_size)]
        return new_input


class EagleMTPProposer(RecurrentEagleMTPProposer):
    """Recurrent Eagle MTP proposer.

    The draft model keeps a cache and repeatedly feeds the previous proposal
    hidden back into the same draft model.
    """

    def propose_next(
        self,
        *,
        main_model_input: ModelInput,
        main_model_output: ModelOutput = None,
        next_token_ids: torch.Tensor,
        b_req_mtp_start_loc: torch.Tensor,
        draft_step: int,
        verify_result=None,
    ) -> SpecProposal:
        assert 0 <= draft_step <= self.backend.mtp_step
        del verify_result
        return self._propose_expanded(
            main_model_input=main_model_input,
            main_model_output=main_model_output,
            next_token_ids=next_token_ids,
            b_req_mtp_start_loc=b_req_mtp_start_loc,
            draft_step=draft_step,
        )

    def _propose_expanded(
        self,
        *,
        main_model_input: ModelInput,
        main_model_output: ModelOutput = None,
        next_token_ids: torch.Tensor,
        b_req_mtp_start_loc: torch.Tensor,
        draft_step: int,
    ) -> SpecProposal:
        del main_model_output
        num_reqs = b_req_mtp_start_loc.shape[0]

        if draft_step == 0:
            eagle_mem_indexes_cpu = None
            eagle_mem_indexes = None
        else:
            eagle_mem_indexes_cpu = self.alloc_extra_mem_indexes(num_reqs * draft_step)
            eagle_mem_indexes = eagle_mem_indexes_cpu.cuda(non_blocking=True)

        draft_model_input = main_model_input
        draft_next_token_ids = next_token_ids
        draft_hidden = self.runtime.get_hidden() if draft_step > 0 else None
        all_next_token_ids = [next_token_ids]
        draft_probs = [] if self.enable_dynamic_mtp else None

        for step in range(draft_step):
            draft_hidden = self.project_draft_decode_hidden(draft_hidden)
            draft_model_input = self.prepare_draft_decode_input(
                model_input=draft_model_input,
                next_token_ids=draft_next_token_ids,
                mtp_draft_input_hiddens=draft_hidden,
            )
            draft_model = self.backend.draft_models[0]
            draft_model_output = draft_model.forward(draft_model_input)
            draft_hidden = self.runtime.get_hidden()

            if self.enable_dynamic_mtp:
                draft_next_token_ids, draft_prob = self.backend._gen_argmax_token_ids_and_prob(draft_model_output)
                draft_probs.append(draft_prob)
            else:
                draft_next_token_ids = self.backend._gen_argmax_token_ids(draft_model_output)

            draft_model_input.b_seq_len += 1
            draft_model_input.max_kv_seq_len += 1
            eagle_mem_indexes_i = eagle_mem_indexes[step * num_reqs : (step + 1) * num_reqs]
            if self.enable_dynamic_mtp:
                from lightllm.server.router.model_infer.mode_backend.update_mem_index import (
                    update_eagle_mem_indexes_triton,
                )

                draft_model_input.mem_indexes = update_eagle_mem_indexes_triton(
                    old_mem_indexes=draft_model_input.mem_indexes,
                    new_step_mem_indexes=eagle_mem_indexes_i,
                    b_req_mtp_start_loc=b_req_mtp_start_loc,
                )
            else:
                draft_model_input.mem_indexes = torch.cat(
                    [
                        draft_model_input.mem_indexes.view(-1, self.backend.mtp_step + 1)[:, 1:],
                        eagle_mem_indexes_i.view(-1, 1),
                    ],
                    dim=1,
                ).view(-1)
            all_next_token_ids.append(draft_next_token_ids)

        return SpecProposal(
            token_ids=torch.stack(all_next_token_ids, dim=1),
            extra_mem_indexes_cpu=eagle_mem_indexes_cpu,
            draft_probs=draft_probs,
        )
