from __future__ import annotations

import torch

from lightllm.common.basemodel.batch_objs import ModelOutput
from lightllm.server.router.model_infer.mtp_speculative.proposers.base import MtpMemIndexesToFree
from lightllm.server.router.model_infer.mtp_speculative.proposers.dflash import DFlashProposer
from lightllm.server.router.model_infer.mtp_speculative.proposers.proposal_type import DFlash2SpecProposal


class DFlash2Proposer(DFlashProposer):
    """Fixed-width DFlash2 block proposer with selector-distribution outputs."""

    def __init__(self, *, backend, enable_dynmaic_mtp: bool) -> None:
        if enable_dynmaic_mtp:
            raise ValueError("DFlash2 does not support dynamic MTP verification")
        super().__init__(backend=backend, enable_dynmaic_mtp=False)

    def _build_proposal(
        self,
        draft_output: ModelOutput,
        req_num: int,
        block_size: int,
        draft_step: int,
        extra_mem_indexes_cpu: torch.Tensor,
    ) -> DFlash2SpecProposal:
        mtp_collector = draft_output.mtp_collector
        selected_token_ids = mtp_collector.draft_token_ids
        candidate_ids = mtp_collector.draft_candidate_ids
        candidate_probs = mtp_collector.draft_candidate_probs
        expected_token_shape = (req_num, block_size - 1)
        assert selected_token_ids.shape == expected_token_shape
        assert candidate_ids.shape == candidate_probs.shape, "candidate id/probability shapes must match"
        assert candidate_ids.shape[:2] == expected_token_shape

        return DFlash2SpecProposal(
            token_ids=selected_token_ids[:, :draft_step].contiguous(),
            extra_mem_indexes_cpu=[MtpMemIndexesToFree(mem_indexes_cpu=extra_mem_indexes_cpu)],
            candidate_ids=candidate_ids[:, :draft_step].contiguous(),
            q_probs=candidate_probs[:, :draft_step].float().contiguous(),
        )
