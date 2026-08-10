from __future__ import annotations

import copy

import torch

from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
from lightllm.server.router.model_infer.speculative.proposers.base import BaseSpecProposer, SpecProposal


class DFlashProposer(BaseSpecProposer):
    """DFlash block proposer aligned to the Eagle3 engine boundary.

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

    @torch.no_grad()
    def build_initial_draft_state(
        self,
        model_input: ModelInput,
        model_output: ModelOutput,
        next_token_ids: torch.Tensor,
    ) -> None:
        target_hidden = model_output.spec_hidden
        assert target_hidden is not None
        if target_hidden.numel() == 0:
            return

        draft_model = self.backend.draft_models[0]
        assert model_input.input_ids is not None
        assert model_input.input_ids.shape[0] == target_hidden.shape[0]
        draft_input = copy.copy(model_input)
        # DFlash consumes target hidden states directly on this prefill path.
        draft_input.mtp_draft_input_hiddens = target_hidden
        draft_model.forward(draft_input)

    @torch.no_grad()
    def propose_next(
        self,
        main_model_input: ModelInput,
        main_model_output: ModelOutput,
        next_token_ids: torch.Tensor,
        b_req_mtp_start_loc: torch.Tensor,
        draft_step: int,
        accept_len: torch.Tensor | None = None,
    ) -> SpecProposal:
        assert 0 <= draft_step <= self.backend.max_draft_step
        assert accept_len is not None, "DFlash proposal requires target accept lengths"

        num_reqs = int(b_req_mtp_start_loc.shape[0])
        draft_model = self.backend.draft_models[0]
        block_size = int(draft_model.block_size)
        assert accept_len.shape[0] == num_reqs
        token_ids = next_token_ids.new_full(
            (next_token_ids.shape[0], draft_step + 1),
            fill_value=1,
        )
        token_ids[:, 0] = next_token_ids

        assert main_model_output is not None and main_model_output.spec_hidden is not None
        self.extend_draft_kv_cache(
            main_model_input=main_model_input,
            target_hidden=main_model_output.spec_hidden,
        )

        if draft_step == 0:
            return SpecProposal(
                token_ids=token_ids,
                extra_mem_indexes_cpu=None,
                draft_probs=None,
            )

        # DFlash drafts from the accepted tail row of each request; one anchor
        # row expands to a complete non-causal draft block.
        selected_rows = self.select_accepted_tail_rows(
            b_req_mtp_start_loc=b_req_mtp_start_loc,
            accept_len=accept_len,
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
        # Standard DFlash has one leading bonus row; DeepSpec checkpoints do not.
        bonus_rows = block_size - draft_step
        assert bonus_rows in (0, 1), (
            f"DFlash block_size={block_size} must equal mtp_step={draft_step} " f"or mtp_step + 1={draft_step + 1}"
        )
        token_ids[selected_rows, 1:] = block_token_ids[:, bonus_rows:]
        return SpecProposal(
            token_ids=token_ids,
            extra_mem_indexes_cpu=draft_mem_indexes_cpu,
            draft_probs=None,
        )

    def extend_draft_kv_cache(self, main_model_input: ModelInput, target_hidden: torch.Tensor) -> None:
        draft_model = self.backend.draft_models[0]
        batch_size = int(target_hidden.shape[0])
        assert batch_size == main_model_input.b_req_idx.shape[0]

        draft_kv_input = copy.copy(main_model_input)
        draft_kv_input.batch_size = batch_size
        draft_kv_input.total_token_num = batch_size
        draft_kv_input.multimodal_params = [{"images": [], "audios": []} for _ in range(batch_size)]
        # This hidden-commit prefill path does not consume token ids, but
        # InferState uses input_ids.shape[0] to build position ids. Keep it
        # aligned with the fixed-shape target verify batch.
        draft_kv_input.input_ids = torch.empty(
            (batch_size,),
            dtype=torch.int64,
            device=target_hidden.device,
        )
        draft_kv_input.max_q_seq_len = 1
        draft_kv_input.prefix_total_token_num = 0
        draft_kv_input.is_prefill = True
        # Match Eagle3's fixed verify commit: write every speculative row and
        # let the accepted-tail sequence length select the valid prefix. Rejected
        # suffix slots are ignored and released by the normal verify free path.
        # Keeping the batch shape static avoids torch.nonzero's implicit D2H
        # synchronization and lets the host enqueue the draft work immediately.
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

    def build_block_draft_input(
        self,
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
        draft_input.draft_step = block_size - 1
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
