from __future__ import annotations

import copy

import torch

from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
from lightllm.server.router.model_infer.speculative.proposers.base import BaseSpecProposer, SpecProposal


class DFlashProposer(BaseSpecProposer):
    """DFlash block proposer aligned to the Eagle3 runtime boundary.

    DFlash remains a non-causal block-prefill draft model, not a recurrent
    token decoder.  The service flow is:
    - verify target tokens
    - extend the DFlash draft KV cache with target hidden rows
    - draft a new non-causal block from the accepted tail row
    The memory lifecycle stays in the normal decode free path:
    - rejected target token slots are freed by the normal decode free path
    - current-block scratch KV uses extra mem slots returned through
      `SpecProposal.extra_mem_indexes_cpu`
    """

    variant = "dflash"

    @torch.no_grad()
    def build_initial_draft_state(
        self,
        *,
        model_input: ModelInput,
        next_token_ids: torch.Tensor,
    ) -> None:
        del next_token_ids
        target_hidden = self.runtime.get_hidden()
        if target_hidden.numel() == 0:
            return

        draft_model = self.backend.draft_models[0]
        assert model_input.input_ids is not None
        assert model_input.input_ids.shape[0] == target_hidden.shape[0]
        draft_input = copy.copy(model_input)
        # DFlash consumes target hidden states directly on this prefill path.
        draft_input.mtp_draft_input_hiddens = target_hidden
        draft_model.forward(draft_input)
        return

    @torch.no_grad()
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
        del main_model_output
        assert 0 <= draft_step <= self.backend.mtp_step
        assert verify_result is not None, "DFlash proposal requires target verify result"

        num_reqs = int(b_req_mtp_start_loc.shape[0])
        draft_model = self.backend.draft_models[0]
        block_size = int(draft_model.block_size)
        assert block_size >= draft_step
        assert verify_result.accept_len.shape[0] == num_reqs
        token_ids = next_token_ids.new_full(
            (next_token_ids.shape[0], draft_step + 1),
            fill_value=1,
        )
        token_ids[:, 0] = next_token_ids

        if draft_step == 0:
            return SpecProposal(
                token_ids=token_ids,
                extra_mem_indexes_cpu=None,
                draft_probs=None,
            )

        self.extend_draft_kv_cache(main_model_input=main_model_input)

        # DFlash only drafts from the accepted tail row of each request.  Unlike
        # MTP, one anchor row expands to a whole non-causal block.
        selected_rows = self.select_accepted_tail_rows(
            b_req_mtp_start_loc=b_req_mtp_start_loc,
            accept_len=verify_result.accept_len,
        )
        draft_input, draft_mem_indexes_cpu = self.build_block_draft_input(
            main_model_input=main_model_input,
            next_token_ids=next_token_ids,
            selected_rows=selected_rows,
            num_reqs=num_reqs,
        )
        draft_model_output = draft_model.forward(draft_input)

        flat_token_ids = self.backend._gen_argmax_token_ids(draft_model_output)
        assert flat_token_ids.numel() == num_reqs * block_size
        block_token_ids = flat_token_ids.reshape(num_reqs, block_size)
        token_ids[selected_rows, 1:] = block_token_ids[:, :draft_step]
        return SpecProposal(
            token_ids=token_ids,
            extra_mem_indexes_cpu=draft_mem_indexes_cpu,
            draft_probs=None,
        )

    def extend_draft_kv_cache(self, *, main_model_input: ModelInput) -> None:
        target_hidden = self.runtime.get_hidden()
        draft_model = self.backend.draft_models[0]
        batch_size = int(target_hidden.shape[0])

        draft_kv_input = copy.copy(main_model_input)
        draft_kv_input.batch_size = batch_size
        draft_kv_input.total_token_num = batch_size
        draft_kv_input.max_q_seq_len = 1
        draft_kv_input.prefix_total_token_num = 0
        draft_kv_input.is_prefill = True
        # Each expanded MTP row writes one target-hidden KV slot for the same request.
        draft_kv_input.b_ready_cache_len = main_model_input.b_seq_len - 1
        draft_kv_input.b_prefill_start_loc = torch.arange(
            batch_size,
            dtype=torch.int32,
            device=target_hidden.device,
        )
        draft_kv_input.b_position_delta = None
        draft_kv_input.b_prefill_has_output_cpu = [False for _ in range(batch_size)]
        draft_kv_input.mtp_draft_input_hiddens = target_hidden
        draft_model.forward(draft_kv_input)
        return

    def build_block_draft_input(
        self,
        *,
        main_model_input: ModelInput,
        next_token_ids: torch.Tensor,
        selected_rows: torch.Tensor,
        num_reqs: int,
    ):
        draft_model = self.backend.draft_models[0]
        num_reqs = int(num_reqs)
        block_size = int(draft_model.block_size)
        assert selected_rows.shape[0] == num_reqs
        draft_mem_indexes_cpu = self.alloc_extra_mem_indexes(num_reqs * block_size)

        draft_input_ids = next_token_ids.new_full(
            (num_reqs * block_size,),
            fill_value=draft_model.mask_token_id,
        )
        # Each block is [accepted_token, mask, ..., mask], matching DeepSpec's
        # draft input.  The proposer later maps the block logits back to
        # [base_token + draft_tokens] for target verification.
        draft_input_ids[::block_size] = next_token_ids.index_select(0, selected_rows)

        block_offsets = torch.arange(
            block_size,
            dtype=main_model_input.b_seq_len.dtype,
            device=next_token_ids.device,
        )
        draft_input = copy.copy(main_model_input)
        draft_input.input_ids = draft_input_ids
        draft_input.total_token_num = draft_input.input_ids.shape[0]
        draft_input.batch_size = draft_input.total_token_num
        draft_input.max_q_seq_len = 1
        draft_input.max_kv_seq_len = main_model_input.max_kv_seq_len + block_size
        draft_input.max_cache_len = draft_input.max_kv_seq_len
        draft_input.b_req_idx = (
            main_model_input.b_req_idx.index_select(0, selected_rows).repeat_interleave(block_size).contiguous()
        )
        draft_input.b_mtp_index = torch.zeros_like(draft_input.b_req_idx)
        # b_seq_len is real metadata, not cosmetic: copy_kv_index_to_req and
        # FA3 use it to place scratch KV and compute the block cache length.
        draft_input.b_seq_len = (
            (main_model_input.b_seq_len.index_select(0, selected_rows)[:, None] + block_offsets[None, :] + 1)
            .reshape(-1)
            .contiguous()
        )
        if main_model_input.b_position_delta is not None:
            draft_input.b_position_delta = (
                main_model_input.b_position_delta.index_select(0, selected_rows)
                .repeat_interleave(block_size)
                .contiguous()
            )
        else:
            draft_input.b_position_delta = torch.zeros_like(draft_input.b_req_idx)
        draft_input.mem_indexes = draft_mem_indexes_cpu.cuda(non_blocking=True)
        draft_input.b_mark_shared_group = torch.zeros_like(draft_input.b_req_idx)
        draft_input.b_mark_shared_group[block_size - 1 :: block_size] = block_size
        draft_input.multimodal_params = [{"images": [], "audios": []} for _ in range(draft_input.batch_size)]
        return draft_input, draft_mem_indexes_cpu
