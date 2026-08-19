from __future__ import annotations

from dataclasses import dataclass

import torch

from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
from lightllm.server.router.model_infer.pin_mem_manager import g_pin_mem_manager
from lightllm.server.router.model_infer.mtp_speculative.proposers.base import (
    BaseSpecProposer,
    MtpMemIndexesToFree,
    SpecProposal,
)
from lightllm.server.router.model_infer.mtp_speculative.proposers.parallel_block_utils import (
    build_parallel_block_draft_input,
    fill_parallel_block_draft_model_kv_state,
    extend_parallel_block_draft_kv_cache,
)


@dataclass
class DSparkSpecProposal(SpecProposal):
    """DSpark proposal with GPU confidence scores and their CPU planner view."""

    schedule_scores: torch.Tensor | None = None
    schedule_scores_cpu: torch.Tensor | None = None


class DSparkProposer(BaseSpecProposer):
    """DSpark semi-autoregressive parallel-block proposer.

    A parallel DFlash-style backbone generates the block features in one pass;
    a lightweight sequential Markov head adds intra-block token dependency.
    The confidence head supplies per-position scheduling scores.
    """

    def fill_draft_model_kv_state(
        self,
        target_model_input: ModelInput,
        target_model_output: ModelOutput,
        target_next_token_ids: torch.Tensor,
    ) -> None:
        fill_parallel_block_draft_model_kv_state(
            proposer=self,
            target_model_input=target_model_input,
            target_model_output=target_model_output,
            target_next_token_ids=target_next_token_ids,
        )

    @torch.no_grad()
    def propose_next(
        self,
        target_model_input: ModelInput,
        target_model_output: ModelOutput,
        target_next_token_ids: torch.Tensor,
        b_req_mtp_start_loc: torch.Tensor,
        draft_step: int,
        accept_len: torch.Tensor | None = None,
    ) -> DSparkSpecProposal:
        request_count = int(b_req_mtp_start_loc.shape[0])
        draft_model = self.backend.draft_models[0]
        block_size = int(draft_model.block_size)
        schedule_scores = None

        accepted_tail_rows = (b_req_mtp_start_loc + accept_len - 1).long()
        draft_input, extra_mem_indexes_cpu = build_parallel_block_draft_input(
            proposer=self,
            target_model_input=target_model_input,
            target_next_token_ids=target_next_token_ids,
            accepted_tail_rows=accepted_tail_rows,
            request_count=request_count,
        )
        extend_parallel_block_draft_kv_cache(
            proposer=self,
            target_model_input=target_model_input,
            target_hidden=target_model_output.mtp_collector.spec_hidden,
        )
        draft_output = draft_model.forward(draft_input)

        if draft_output.mtp_collector.draft_token_ids is None:
            flat_draft_token_ids = self.backend._gen_argmax_token_ids(draft_output)
        else:
            flat_draft_token_ids = draft_output.mtp_collector.draft_token_ids
        block_draft_token_ids = flat_draft_token_ids.reshape(request_count, block_size)
        proposal_token_ids = block_draft_token_ids[:, :draft_step].contiguous()

        if self.enable_dynmaic_mtp:
            confidence_logits = draft_output.mtp_collector.confidence_logits
            if confidence_logits is None:
                raise RuntimeError("DSpark dynamic verify requires confidence head logits")
            # Match the clamp used by the GPU dynamic row selector before it
            # converts conditional confidence to prefix survival probability.
            schedule_scores = (
                confidence_logits[:, :draft_step]
                .sigmoid()
                .clamp(
                    min=0.01,
                    max=0.99,
                )
                .contiguous()
            )

        schedule_scores_cpu = None
        if schedule_scores is not None:
            schedule_scores_cpu = g_pin_mem_manager.async_copy_from_gpu_tensor(
                key="dspark_confidence_probs",
                gpu_tensor=schedule_scores,
            )

        return DSparkSpecProposal(
            token_ids=proposal_token_ids,
            extra_mem_indexes_cpu=[MtpMemIndexesToFree(mem_indexes_cpu=extra_mem_indexes_cpu)],
            schedule_scores=schedule_scores,
            schedule_scores_cpu=schedule_scores_cpu,
        )
