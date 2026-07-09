from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from lightllm.common.basemodel.triton_kernel.mtp_utils import mtp_scatter_next_token_ids, mtp_verify


@dataclass
class SpecVerifyResult:
    """GPU verification result produced after target decode.

    `accept_len` is per logical request and includes the target token at
    position 0:

        accept_len: [logical_req_num]

    `accepted_index` is per verified row in the target batch.  It marks rows
    whose token should be committed and post-processed:

        accepted_index: [verify_batch]
    """

    accept_len: torch.Tensor
    accepted_index: torch.Tensor


class SpecVerifier:
    """Service verifier for LightLLM MTP layout.

    DeepSpec verifies by running target logits over
    [current_token + draft_tokens] and applying rejection sampling.  LightLLM's
    service path stores pending candidates in req_sampling_params_manager and
    uses Triton kernels to:
    - compare the target sampled tokens with previously scattered candidates
    - compute per-request accepted prefix length
    - scatter the newly proposed candidates for the next decode iteration
    """

    def __init__(self, backend) -> None:
        self.backend = backend

    def verify_target_tokens(
        self,
        *,
        new_next_token_ids: torch.Tensor,
        b_req_idx: torch.Tensor,
        b_req_mtp_start_loc: torch.Tensor,
    ) -> SpecVerifyResult:
        """Verify target sampled ids against previously proposed ids.

        Inputs:
        - `new_next_token_ids`: target sampled ids, shape [verify_batch]
        - `b_req_idx`: request ids for each target row, shape [verify_batch]
        - `b_req_mtp_start_loc`: first row of each logical request in the
          MTP-expanded target batch, shape [logical_req_num]
        """

        mtp_accept_len, accepted_index = mtp_verify(
            req_to_next_token_ids=self.backend.model.req_manager.req_sampling_params_manager.req_to_next_token_ids,
            b_req_mtp_start_loc=b_req_mtp_start_loc,
            new_next_token_ids=new_next_token_ids,
            b_req_idx=b_req_idx,
        )
        return SpecVerifyResult(accept_len=mtp_accept_len, accepted_index=accepted_index)

    def scatter_next_tokens(
        self,
        *,
        b_req_mtp_start_loc: torch.Tensor,
        all_next_token_ids: torch.Tensor,
        b_req_idx: torch.Tensor,
        mtp_accept_len: torch.Tensor,
        all_next_token_probs: Optional[torch.Tensor] = None,
    ) -> None:
        """Scatter target+draft candidates into per-request next-token buffers.

        Inputs:
        - `all_next_token_ids`: [verify_batch, mtp_step + 1].  Column 0 is the
          target sampled token from this iteration; remaining columns are draft
          candidates padded to the static MTP width when dynamic MTP produces a
          shorter proposal.
        - `all_next_token_probs`: optional [verify_batch, mtp_step + 1].
          Current dynamic MTP stores selected-token probabilities, not full
          vocab distributions.
        """

        mtp_scatter_next_token_ids(
            req_to_next_token_ids=self.backend.model.req_manager.req_sampling_params_manager.req_to_next_token_ids,
            b_req_mtp_start_loc=b_req_mtp_start_loc,
            all_next_token_ids=all_next_token_ids,
            b_req_idx=b_req_idx,
            mtp_accept_len=mtp_accept_len,
            req_to_next_token_probs=self.backend.model.req_manager.req_sampling_params_manager.req_to_next_token_probs,
            all_next_token_probs=all_next_token_probs,
        )
        return
