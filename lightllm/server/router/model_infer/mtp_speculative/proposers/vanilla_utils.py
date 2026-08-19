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


def build_chained_mtp_draft_state_from_prefill(
    proposer: BaseSpecProposer,
    target_model_input: ModelInput,
    target_model_output: ModelOutput,
    next_token_ids: torch.Tensor,
) -> None:
    """构建 Vanilla chained MTP 各级 draft model 的 prefill state。"""

    from lightllm.server.router.model_infer.mode_backend.mtp_pre_process import prepare_mtp_prefill_inputs

    draft_hidden = target_model_output.mtp_collector.spec_hidden
    draft_token_ids = next_token_ids
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
    draft_step: int,
) -> VanillaSpecProposal:
    """依次运行 Vanilla chained MTP 模块并生成 proposal。"""

    verify_row_count = int(next_token_ids.shape[0])
    draft_token_ids = next_token_ids
    draft_hidden = main_model_output.mtp_collector.spec_hidden
    proposal_token_ids = next_token_ids.new_empty((verify_row_count, draft_step + 1))
    proposal_token_ids[:, 0] = next_token_ids
    schedule_scores = (
        torch.empty(
            (verify_row_count, draft_step),
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
            schedule_scores[:, step] = draft_token_probs
        else:
            draft_token_ids = proposer.backend._gen_argmax_token_ids(draft_output)
        proposal_token_ids[:, step + 1] = draft_token_ids

    return VanillaSpecProposal(
        token_ids=proposal_token_ids,
        extra_mem_indexes_cpu=[],
        schedule_scores=schedule_scores,
    )
