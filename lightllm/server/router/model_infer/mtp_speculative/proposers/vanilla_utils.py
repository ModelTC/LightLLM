"""Vanilla chained MTP proposer 共享辅助函数。"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
from lightllm.server.router.model_infer.mtp_speculative.proposers.base import BaseSpecProposer, SpecProposal


@dataclass
class VanillaSpecProposal(SpecProposal):
    """Vanilla proposal with optional selected-token probabilities."""

    schedule_scores: torch.Tensor | None = None


def fill_chained_mtp_draft_model_kv_state(
    proposer: BaseSpecProposer,
    target_model_input: ModelInput,
    target_model_output: ModelOutput,
    target_next_token_ids: torch.Tensor,
) -> None:
    """构建 Vanilla chained MTP 各级 draft model 的 prefill state。"""

    from lightllm.server.router.model_infer.mode_backend.mtp_pre_process import prepare_mtp_prefill_inputs

    draft_hidden = target_model_output.mtp_collector.spec_hidden
    draft_token_ids = target_next_token_ids
    for draft_model in proposer.backend.draft_models:
        prepare_mtp_prefill_inputs(
            model_input=target_model_input,
            b_next_token_ids=draft_token_ids,
            mtp_draft_input_hiddens=draft_hidden,
        )
        draft_output = draft_model.forward(target_model_input)
        draft_hidden = draft_output.mtp_collector.spec_hidden
        draft_token_ids = proposer.backend._gen_argmax_token_ids(draft_output)


def propose_next_chained_mtp(
    proposer: BaseSpecProposer,
    main_model_input: ModelInput,
    main_model_output: ModelOutput,
    next_token_ids: torch.Tensor,
    b_req_mtp_start_loc: torch.Tensor,
    draft_step: int,
    accept_len: torch.Tensor,
) -> VanillaSpecProposal:
    """依次运行 Vanilla chained MTP 模块并生成 proposal。"""

    request_count = int(b_req_mtp_start_loc.shape[0])
    accepted_tail_rows = (b_req_mtp_start_loc + accept_len - 1).long()
    draft_token_ids = next_token_ids
    draft_hidden = main_model_output.mtp_collector.spec_hidden
    proposal_token_ids = next_token_ids.new_empty((request_count, draft_step))
    schedule_scores = (
        torch.empty(
            (request_count, draft_step),
            dtype=torch.float32,
            device=next_token_ids.device,
        )
        if proposer.enable_dynmaic_mtp
        else None
    )

    for step in range(draft_step):
        draft_model = proposer.backend.draft_models[step]
        main_model_input.input_ids = draft_token_ids
        main_model_input.mtp_draft_input_hiddens = draft_hidden
        draft_output = draft_model.forward(main_model_input)
        draft_hidden = draft_output.mtp_collector.spec_hidden
        if proposer.enable_dynmaic_mtp:
            draft_token_ids, draft_token_probs = proposer.backend._gen_argmax_token_ids_and_prob(draft_output)
            schedule_scores[:, step] = draft_token_probs.index_select(0, accepted_tail_rows)
        else:
            draft_token_ids = proposer.backend._gen_argmax_token_ids(draft_output)
        proposal_token_ids[:, step] = draft_token_ids.index_select(0, accepted_tail_rows)

    return VanillaSpecProposal(
        token_ids=proposal_token_ids,
        extra_mem_indexes_cpu=[],
        schedule_scores=schedule_scores,
    )
