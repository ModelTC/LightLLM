"""DP overlap EAGLE proposer 共享辅助函数。"""

from __future__ import annotations

from typing import Callable

import torch

from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
from lightllm.server.router.model_infer.mtp_speculative import utils as mtp_utils
from lightllm.server.router.model_infer.mtp_speculative.dp_overlap_proposers.base import BaseDpOverlapProposer
from lightllm.server.router.model_infer.mtp_speculative.proposers.base import MtpMemIndexesToFree
from lightllm.server.router.model_infer.mtp_speculative.proposers.eagle_utils import (
    EagleSpecProposal,
    generate_eagle_token_ids,
    prepare_eagle_verify_extend_input,
)


def fill_dp_eagle_draft_model_kv_state_overlap(
    proposer: BaseDpOverlapProposer,
    target_model_input0: ModelInput,
    target_model_output0: ModelOutput,
    target_next_token_ids0: torch.Tensor,
    target_model_input1: ModelInput,
    target_model_output1: ModelOutput,
    target_next_token_ids1: torch.Tensor,
) -> None:
    """使用两个 target prefill microbatch 初始化 EAGLE draft state。"""

    from lightllm.server.router.model_infer.mode_backend.mtp_pre_process import prepare_mtp_prefill_inputs

    prepare_mtp_prefill_inputs(
        model_input=target_model_input0,
        b_next_token_ids=target_next_token_ids0,
        mtp_draft_input_hiddens=target_model_output0.mtp_collector.spec_hidden,
    )
    prepare_mtp_prefill_inputs(
        model_input=target_model_input1,
        b_next_token_ids=target_next_token_ids1,
        mtp_draft_input_hiddens=target_model_output1.mtp_collector.spec_hidden,
    )
    proposer.backend.draft_models[0].microbatch_overlap_prefill(target_model_input0, target_model_input1)


def pad_dp_step_mem_indexes(
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


def propose_next_dp_eagle_autoregressive_overlap(
    proposer: BaseDpOverlapProposer,
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
    map_draft_token_ids: Callable[[torch.Tensor], torch.Tensor],
) -> EagleSpecProposal:
    """运行 DP EAGLE extend 后接单 token overlap decode 的 proposal 流程。"""

    verify_width = proposer.backend.max_draft_step + 1
    model_inputs = (target_model_input0, target_model_input1)
    model_outputs = (target_model_output0, target_model_output1)
    target_next_token_ids_by_batch = (target_next_token_ids0, target_next_token_ids1)
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
        target_next_token_ids_by_batch,
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
        prepare_eagle_verify_extend_input(
            model_input=model_input,
            input_ids=token_ids,
            target_hidden=model_output.mtp_collector.spec_hidden,
        )

    total_real_request_count = sum(real_request_counts)
    proposal_token_ids = target_next_token_ids0.new_empty((total_real_request_count, draft_step))

    draft_model = proposer.backend.draft_models[0]
    extend_outputs = draft_model.microbatch_overlap_prefill(*model_inputs)

    draft_token_ids_by_batch = []
    draft_hiddens_by_batch = []
    draft_seq_lens_by_batch = []
    draft_req_indices_by_batch = []
    proposal_row_offsets = (0, real_request_counts[0])
    for batch_index, (model_input, extend_output, accepted_tail_rows, real_request_count) in enumerate(
        zip(model_inputs, extend_outputs, accepted_tail_rows_by_batch, real_request_counts)
    ):
        accepted_tail_output = ModelOutput(logits=extend_output.logits.index_select(0, accepted_tail_rows))
        draft_token_ids = generate_eagle_token_ids(proposer, accepted_tail_output, map_draft_token_ids)
        draft_token_ids_by_batch.append(draft_token_ids)
        draft_hiddens_by_batch.append(extend_output.mtp_collector.spec_hidden.index_select(0, accepted_tail_rows))
        draft_seq_lens_by_batch.append(model_input.b_seq_len.index_select(0, accepted_tail_rows) + 1)
        draft_req_indices_by_batch.append(model_input.b_req_idx.index_select(0, accepted_tail_rows))
        proposal_row_start = proposal_row_offsets[batch_index]
        proposal_token_ids[proposal_row_start : proposal_row_start + real_request_count, 0] = draft_token_ids[
            :real_request_count
        ]

    if draft_step == 1:
        return EagleSpecProposal(
            token_ids=proposal_token_ids,
            extra_mem_indexes_cpu=[],
            schedule_scores=None,
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

    extra_mem_indexes_cpu = mtp_utils.alloc_mem_indexes(total_real_request_count * (draft_step - 1))
    extra_mem_indexes = extra_mem_indexes_cpu.to(device=target_next_token_ids0.device, non_blocking=True)
    hold_mem_index = proposer.backend.model.req_manager.mem_manager.HOLD_TOKEN_MEMINDEX

    for step in range(1, draft_step):
        mem_start = (step - 1) * total_real_request_count
        step_mem_indexes = extra_mem_indexes[mem_start : mem_start + total_real_request_count]
        real_mem_start = 0
        for batch_index, model_input in enumerate(model_inputs):
            real_request_count = real_request_counts[batch_index]
            real_mem_indexes = step_mem_indexes[real_mem_start : real_mem_start + real_request_count]
            real_mem_start += real_request_count
            model_input.input_ids = draft_token_ids_by_batch[batch_index]
            model_input.mtp_draft_input_hiddens = draft_hiddens_by_batch[batch_index]
            model_input.mem_indexes = pad_dp_step_mem_indexes(
                real_mem_indexes=real_mem_indexes,
                request_capacity=request_capacities_by_batch[batch_index],
                hold_mem_index=hold_mem_index,
            )
            model_input.max_kv_seq_len = max_kv_seq_lens_by_batch[batch_index] + step
            model_input.total_token_num = model_input.batch_size * model_input.max_kv_seq_len

        draft_outputs = draft_model.microbatch_overlap_decode(*model_inputs)
        for batch_index, draft_output in enumerate(draft_outputs):
            draft_token_ids = generate_eagle_token_ids(proposer, draft_output, map_draft_token_ids)
            draft_token_ids_by_batch[batch_index] = draft_token_ids
            draft_hiddens_by_batch[batch_index] = draft_output.mtp_collector.spec_hidden
            draft_seq_lens_by_batch[batch_index].add_(1)
            real_request_count = real_request_counts[batch_index]
            proposal_row_start = proposal_row_offsets[batch_index]
            proposal_token_ids[proposal_row_start : proposal_row_start + real_request_count, step] = draft_token_ids[
                :real_request_count
            ]

    return EagleSpecProposal(
        token_ids=proposal_token_ids,
        extra_mem_indexes_cpu=[MtpMemIndexesToFree(mem_indexes_cpu=extra_mem_indexes_cpu)],
        schedule_scores=None,
    )


def propose_next_dp_eagle_fixed_layout_overlap(
    proposer: BaseDpOverlapProposer,
    target_model_input0: ModelInput,
    target_model_output0: ModelOutput,
    target_next_token_ids0: torch.Tensor,
    real_verify_rows0: int,
    accept_len0: torch.Tensor,
    target_model_input1: ModelInput,
    target_model_output1: ModelOutput,
    target_next_token_ids1: torch.Tensor,
    real_verify_rows1: int,
    accept_len1: torch.Tensor,
    draft_step: int,
    map_draft_token_ids: Callable[[torch.Tensor], torch.Tensor],
) -> EagleSpecProposal:
    """以 expanded verify-row layout 运行 decode，返回按真实请求压缩的 proposal。"""

    verify_width = proposer.backend.max_draft_step + 1
    model_inputs = (target_model_input0, target_model_input1)
    real_verify_row_counts = (int(real_verify_rows0), int(real_verify_rows1))
    real_request_counts = tuple(row_count // verify_width for row_count in real_verify_row_counts)
    request_capacities_by_batch = tuple(model_input.batch_size // verify_width for model_input in model_inputs)
    total_real_request_count = sum(real_request_counts)

    proposal_token_ids = target_next_token_ids0.new_empty((total_real_request_count, draft_step))
    proposal_row_offsets = (0, real_request_counts[0])
    accepted_tail_rows_by_batch = (
        torch.arange(0, target_model_input0.batch_size, verify_width, device=target_next_token_ids0.device)
        + accept_len0
        - 1,
        torch.arange(0, target_model_input1.batch_size, verify_width, device=target_next_token_ids1.device)
        + accept_len1
        - 1,
    )

    extra_mem_indexes_cpu = mtp_utils.alloc_mem_indexes(total_real_request_count * draft_step)
    extra_mem_indexes = extra_mem_indexes_cpu.to(device=target_next_token_ids0.device, non_blocking=True)
    split = real_request_counts[0] * draft_step
    extra_mem_indexes_by_batch = (
        extra_mem_indexes[:split],
        extra_mem_indexes[split:],
    )

    draft_token_ids_by_batch = [target_next_token_ids0, target_next_token_ids1]
    draft_hiddens_by_batch = [
        target_model_output0.mtp_collector.spec_hidden,
        target_model_output1.mtp_collector.spec_hidden,
    ]
    draft_model = proposer.backend.draft_models[0]
    hold_mem_index = proposer.backend.model.req_manager.mem_manager.HOLD_TOKEN_MEMINDEX

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
            step_mem_indexes = pad_dp_step_mem_indexes(
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

            draft_token_ids_by_batch[batch_index] = generate_eagle_token_ids(
                proposer,
                draft_output,
                map_draft_token_ids,
            )
            draft_hiddens_by_batch[batch_index] = draft_output.mtp_collector.spec_hidden

        for batch_index, draft_token_ids in enumerate(draft_token_ids_by_batch):
            real_request_count = real_request_counts[batch_index]
            proposal_row_start = proposal_row_offsets[batch_index]
            proposal_token_ids[
                proposal_row_start : proposal_row_start + real_request_count, step
            ] = draft_token_ids.index_select(
                0,
                accepted_tail_rows_by_batch[batch_index][:real_request_count].long(),
            )

    return EagleSpecProposal(
        token_ids=proposal_token_ids,
        extra_mem_indexes_cpu=[MtpMemIndexesToFree(mem_indexes_cpu=extra_mem_indexes_cpu)],
        schedule_scores=None,
    )
