from __future__ import annotations


from lightllm.common.basemodel.batch_objs import ModelOutput
from lightllm.server.router.model_infer.mtp_speculative.proposers.dflash import DFlashProposer
from lightllm.server.router.model_infer.mtp_speculative.proposers.proposal_type import DFlashSpecProposal


class DFlash2Proposer(DFlashProposer):
    """Full-block greedy drafting with optional dynamic target verification."""

    def _build_proposal(
        self,
        draft_output: ModelOutput,
        req_num: int,
        block_size: int,
        draft_step: int,
    ) -> DFlashSpecProposal:
        mtp_collector = draft_output.mtp_collector
        selected_token_ids = mtp_collector.draft_token_ids
        assert selected_token_ids.shape == (req_num, block_size - 1)
        assert 0 < draft_step <= block_size - 1

        schedule_scores = None
        if self.enable_dynmaic_mtp:
            confidence_logits = mtp_collector.confidence_logits
            if confidence_logits is None:
                raise RuntimeError("DFlash2 dynamic verify requires selector confidence logits")
            assert confidence_logits.shape == selected_token_ids.shape
            schedule_scores = confidence_logits[:, :draft_step].float().sigmoid().clamp(0.01, 0.99).contiguous()

        return DFlashSpecProposal(
            token_ids=selected_token_ids[:, :draft_step].contiguous(),
            schedule_scores=schedule_scores,
        )
