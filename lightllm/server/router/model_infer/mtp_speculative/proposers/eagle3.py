from __future__ import annotations

import torch

from lightllm.server.router.model_infer.mtp_speculative.proposers.eagle_mtp import AutoregressiveEagleProposer


class Eagle3Proposer(AutoregressiveEagleProposer):
    """Eagle3 proposer.

    It uses the shared autoregressive proposal flow with an additional
    draft-to-target vocabulary mapping step.
    """

    def _map_draft_token_ids(self, draft_token_ids: torch.Tensor) -> torch.Tensor:
        return self.backend.draft_models[0].map_draft_vocab_to_main_vocab(draft_token_ids)
