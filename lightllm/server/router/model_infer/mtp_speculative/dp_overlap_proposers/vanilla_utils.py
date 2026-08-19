"""DP overlap Vanilla proposer 共享辅助函数。"""

from __future__ import annotations

import torch

from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
from lightllm.server.router.model_infer.mtp_speculative.dp_overlap_proposers.base import BaseDpOverlapProposer
from lightllm.server.router.model_infer.mtp_speculative.proposers.vanilla_utils import VanillaSpecProposal


def build_dp_chained_mtp_draft_state_from_prefill_overlap(
    proposer: BaseDpOverlapProposer,
    target_model_input0: ModelInput,
    target_model_output0: ModelOutput,
    next_token_ids0: torch.Tensor,
    target_model_input1: ModelInput,
    target_model_output1: ModelOutput,
    next_token_ids1: torch.Tensor,
) -> None:
    """为两个 DP microbatch 依次构建 Vanilla chained draft state。"""

    from lightllm.server.router.model_infer.mode_backend.mtp_pre_process import prepare_mtp_prefill_inputs

    model_inputs = (target_model_input0, target_model_input1)
    draft_hiddens = [
        target_model_output0.mtp_collector.spec_hidden,
        target_model_output1.mtp_collector.spec_hidden,
    ]
    draft_token_ids = [next_token_ids0, next_token_ids1]

    for draft_model in proposer.backend.draft_models:
        for batch_index, model_input in enumerate(model_inputs):
            prepare_mtp_prefill_inputs(
                model_input=model_input,
                b_next_token_ids=draft_token_ids[batch_index],
                mtp_draft_input_hiddens=draft_hiddens[batch_index],
            )
        draft_outputs = draft_model.microbatch_overlap_prefill(*model_inputs)
        for batch_index, draft_output in enumerate(draft_outputs):
            draft_hiddens[batch_index] = draft_output.mtp_collector.spec_hidden
            draft_token_ids[batch_index] = proposer.backend._gen_argmax_token_ids(draft_output)


def propose_next_dp_chained_mtp_overlap(
    proposer: BaseDpOverlapProposer,
    main_model_input0: ModelInput,
    main_model_output0: ModelOutput,
    next_token_ids0: torch.Tensor,
    real_verify_rows0: int,
    main_model_input1: ModelInput,
    main_model_output1: ModelOutput,
    next_token_ids1: torch.Tensor,
    real_verify_rows1: int,
    draft_step: int,
) -> VanillaSpecProposal:
    """为两个 DP microbatch 运行 Vanilla chained overlap decode。"""

    model_inputs = (main_model_input0, main_model_input1)
    real_verify_rows = (int(real_verify_rows0), int(real_verify_rows1))
    draft_token_ids = [next_token_ids0, next_token_ids1]
    draft_hiddens = [
        main_model_output0.mtp_collector.spec_hidden,
        main_model_output1.mtp_collector.spec_hidden,
    ]
    proposal_token_ids = next_token_ids0.new_empty((sum(real_verify_rows), draft_step + 1))
    proposal_token_ids[:real_verify_rows0, 0] = next_token_ids0[:real_verify_rows0]
    proposal_token_ids[real_verify_rows0:, 0] = next_token_ids1[:real_verify_rows1]

    for step in range(draft_step):
        for batch_index, model_input in enumerate(model_inputs):
            model_input.input_ids = draft_token_ids[batch_index]
            model_input.mtp_draft_input_hiddens = draft_hiddens[batch_index]

        draft_outputs = proposer.backend.draft_models[step].microbatch_overlap_decode(*model_inputs)
        for batch_index, draft_output in enumerate(draft_outputs):
            draft_hiddens[batch_index] = draft_output.mtp_collector.spec_hidden
            draft_token_ids[batch_index] = proposer.backend._gen_argmax_token_ids(draft_output)

        proposal_token_ids[:real_verify_rows0, step + 1] = draft_token_ids[0][:real_verify_rows0]
        proposal_token_ids[real_verify_rows0:, step + 1] = draft_token_ids[1][:real_verify_rows1]

    return VanillaSpecProposal(
        token_ids=proposal_token_ids,
        extra_mem_indexes_cpu=[],
        schedule_scores=None,
    )
