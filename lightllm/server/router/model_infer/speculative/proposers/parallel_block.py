from __future__ import annotations

import copy

import torch

from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
from lightllm.server.router.model_infer.speculative.proposers.base import BaseSpecProposer


class ParallelBlockProposer(BaseSpecProposer):
    """Shared state and input preparation for parallel block drafters.

    DFlash and DSpark consume verified target output, extend the drafter KV
    cache with target hidden rows, then generate a new block in one parallel
    backbone forward.
    """

    def get_draft_steps(self):
        return (self.backend.max_draft_step,)

    def get_draft_cost_ms(
        self,
        draft_infer_costs,
        req_num: int,
        verify_batch_size: int,
        draft_step: int,
    ) -> float:
        block_size = self.backend.draft_models[0].block_size
        # One forward commits verified rows; another generates one complete
        # checkpoint-defined block per request.
        extend_cost_ms = draft_infer_costs.estimate(verify_batch_size)
        block_cost_ms = draft_infer_costs.get(req_num * block_size)
        return extend_cost_ms + block_cost_ms

    @torch.no_grad()
    def build_draft_state_from_prefill(
        self,
        target_model_input: ModelInput,
        target_model_output: ModelOutput,
        next_token_ids: torch.Tensor,
    ) -> None:
        target_hidden = target_model_output.spec_hidden
        if target_hidden.numel() == 0:
            return

        # Parallel block drafters consume target hidden states directly.
        target_model_input.mtp_draft_input_hiddens = target_hidden
        self.backend.draft_models[0].forward(target_model_input)

    def extend_draft_kv_cache(self, main_model_input: ModelInput, target_hidden: torch.Tensor) -> None:
        # Target decode has finished; reuse its row-aligned input for draft KV commit.
        main_model_input.total_token_num = main_model_input.batch_size
        main_model_input.prefix_total_token_num = 0
        main_model_input.is_prefill = True
        main_model_input.b_ready_cache_len = main_model_input.b_seq_len - 1
        main_model_input.b_prefill_start_loc = torch.arange(
            main_model_input.batch_size,
            dtype=torch.int32,
            device=target_hidden.device,
        )
        main_model_input.mtp_draft_input_hiddens = target_hidden
        self.backend.draft_models[0].forward(main_model_input)

    def build_block_draft_input(
        self,
        main_model_input: ModelInput,
        next_token_ids: torch.Tensor,
        accepted_tail_rows: torch.Tensor,
        request_count: int,
    ):
        draft_model = self.backend.draft_models[0]
        block_size = int(draft_model.block_size)
        extra_mem_indexes_cpu = self.alloc_extra_mem_indexes(request_count * block_size)

        block_input_ids = next_token_ids.new_full(
            (request_count * block_size,),
            fill_value=draft_model.mask_token_id,
        )
        # Block input layout: [accepted token, mask, ..., mask]. The block
        # logits become [base token + draft tokens] for target verification.
        block_input_ids[::block_size] = next_token_ids.index_select(0, accepted_tail_rows)

        block_offsets = torch.arange(
            block_size,
            dtype=main_model_input.b_seq_len.dtype,
            device=next_token_ids.device,
        )
        draft_input = copy.copy(main_model_input)
        draft_input.input_ids = block_input_ids
        draft_input.total_token_num = draft_input.input_ids.shape[0]
        draft_input.batch_size = draft_input.total_token_num
        draft_input.max_q_seq_len = 1
        draft_input.max_kv_seq_len = main_model_input.max_kv_seq_len + block_size
        draft_input.draft_step = block_size - 1
        draft_input.b_req_idx = (
            main_model_input.b_req_idx.index_select(0, accepted_tail_rows).repeat_interleave(block_size).contiguous()
        )
        draft_input.b_mtp_index = torch.zeros_like(draft_input.b_req_idx)
        # copy_kv_index_to_req and FA3 use these lengths to place scratch KV and
        # compute the block cache length.
        draft_input.b_seq_len = (
            (main_model_input.b_seq_len.index_select(0, accepted_tail_rows)[:, None] + block_offsets[None, :] + 1)
            .reshape(-1)
            .contiguous()
        )
        # Position delta is request-level metadata shared by every block row.
        draft_input.b_position_delta = (
            main_model_input.b_position_delta.index_select(0, accepted_tail_rows)
            .repeat_interleave(block_size)
            .contiguous()
        )
        draft_input.mem_indexes = extra_mem_indexes_cpu.to(device=next_token_ids.device, non_blocking=True)
        draft_input.b_mark_shared_group = torch.zeros_like(draft_input.b_req_idx)
        draft_input.b_mark_shared_group[block_size - 1 :: block_size] = block_size
        empty_multimodal_params = {"images": [], "audios": []}
        draft_input.multimodal_params = [empty_multimodal_params] * draft_input.batch_size
        return draft_input, extra_mem_indexes_cpu
