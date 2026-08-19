"""EAGLE proposer 共享辅助函数。"""

from __future__ import annotations

import copy
from typing import Callable

import torch

from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
from lightllm.server.router.model_infer.mtp_speculative import utils as mtp_utils
from lightllm.server.router.model_infer.mtp_speculative.proposers.base import BaseSpecProposer, SpecProposal


def build_eagle_draft_state_from_prefill(
    proposer: BaseSpecProposer,
    target_model_input: ModelInput,
    target_model_output: ModelOutput,
    next_token_ids: torch.Tensor,
) -> None:
    """使用 target prefill 输出初始化 EAGLE draft state。"""

    from lightllm.server.router.model_infer.mode_backend.mtp_pre_process import prepare_mtp_prefill_inputs

    prepare_mtp_prefill_inputs(
        model_input=target_model_input,
        b_next_token_ids=next_token_ids,
        mtp_draft_input_hiddens=target_model_output.mtp_collector.spec_hidden,
    )
    proposer.backend.draft_models[0].forward(target_model_input)


def generate_eagle_token_ids(
    proposer: BaseSpecProposer,
    model_output: ModelOutput,
    map_draft_token_ids: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    return map_draft_token_ids(proposer.backend._gen_argmax_token_ids(model_output))


def generate_eagle_token_ids_and_prob(
    proposer: BaseSpecProposer,
    model_output: ModelOutput,
    map_draft_token_ids: Callable[[torch.Tensor], torch.Tensor],
):
    draft_token_ids, draft_token_probs = proposer.backend._gen_argmax_token_ids_and_prob(model_output)
    return map_draft_token_ids(draft_token_ids), draft_token_probs


def prepare_eagle_verify_extend_input(
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


def propose_next_eagle(
    proposer: BaseSpecProposer,
    main_model_input: ModelInput,
    main_model_output: ModelOutput,
    next_token_ids: torch.Tensor,
    b_req_mtp_start_loc: torch.Tensor,
    draft_step: int,
    accept_len: torch.Tensor | None,
    map_draft_token_ids: Callable[[torch.Tensor], torch.Tensor],
) -> SpecProposal:
    """运行 EAGLE extend 后接单 token decode 的通用 proposal 流程。"""

    verify_row_count = int(next_token_ids.shape[0])
    request_count = int(b_req_mtp_start_loc.shape[0])
    proposal_token_ids = next_token_ids.new_full(
        (verify_row_count, draft_step + 1),
        fill_value=1,
    )
    proposal_token_ids[:, 0].copy_(next_token_ids)
    collect_schedule_scores = proposer.enable_dynmaic_mtp
    schedule_scores = (
        torch.zeros(
            (verify_row_count, draft_step),
            dtype=torch.float32,
            device=next_token_ids.device,
        )
        if collect_schedule_scores
        else None
    )
    if draft_step == 0:
        return SpecProposal(
            token_ids=proposal_token_ids,
            extra_mem_indexes_cpu=None,
            schedule_scores=schedule_scores,
        )

    accepted_tail_rows = (b_req_mtp_start_loc + accept_len - 1).long()
    draft_model = proposer.backend.draft_models[0]
    position_delta = main_model_input.b_position_delta
    prepare_eagle_verify_extend_input(
        model_input=main_model_input,
        input_ids=next_token_ids,
        target_hidden=main_model_output.mtp_collector.spec_hidden,
    )
    extend_output = draft_model.forward(main_model_input)

    accepted_tail_output = ModelOutput(logits=extend_output.logits.index_select(0, accepted_tail_rows))
    if collect_schedule_scores:
        draft_token_ids, draft_token_probs = generate_eagle_token_ids_and_prob(
            proposer=proposer,
            model_output=accepted_tail_output,
            map_draft_token_ids=map_draft_token_ids,
        )
        schedule_scores[accepted_tail_rows, 0] = draft_token_probs.float()
    else:
        draft_token_ids = generate_eagle_token_ids(
            proposer=proposer,
            model_output=accepted_tail_output,
            map_draft_token_ids=map_draft_token_ids,
        )
    proposal_token_ids[accepted_tail_rows, 1] = draft_token_ids
    draft_hidden = extend_output.mtp_collector.spec_hidden.index_select(0, accepted_tail_rows)

    if draft_step == 1:
        return SpecProposal(
            token_ids=proposal_token_ids,
            extra_mem_indexes_cpu=None,
            schedule_scores=schedule_scores,
        )

    extra_mem_indexes_cpu = mtp_utils.alloc_mem_indexes(request_count * (draft_step - 1))
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
            draft_token_ids, draft_token_probs = generate_eagle_token_ids_and_prob(
                proposer=proposer,
                model_output=draft_output,
                map_draft_token_ids=map_draft_token_ids,
            )
            schedule_scores[accepted_tail_rows, step] = draft_token_probs.float()
        else:
            draft_token_ids = generate_eagle_token_ids(
                proposer=proposer,
                model_output=draft_output,
                map_draft_token_ids=map_draft_token_ids,
            )
        proposal_token_ids[accepted_tail_rows, step + 1] = draft_token_ids
        draft_hidden = draft_output.mtp_collector.spec_hidden
        draft_seq_lens.add_(1)

    return SpecProposal(
        token_ids=proposal_token_ids,
        extra_mem_indexes_cpu=extra_mem_indexes_cpu,
        schedule_scores=schedule_scores,
    )
