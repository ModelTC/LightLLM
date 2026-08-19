from abc import ABC, abstractmethod

import torch

from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
from lightllm.server.router.model_infer.mtp_speculative.proposers.base import BaseSpecProposer, SpecProposal


class BaseDpOverlapProposer(BaseSpecProposer, ABC):
    """DP proposer 的完整接口，扩展双 microbatch overlap 操作。"""

    @abstractmethod
    def fill_draft_model_kv_state_overlap(
        self,
        target_model_input0: ModelInput,
        target_model_output0: ModelOutput,
        target_next_token_ids0: torch.Tensor,
        target_model_input1: ModelInput,
        target_model_output1: ModelOutput,
        target_next_token_ids1: torch.Tensor,
    ) -> None:
        """Build draft state from two overlapped target-prefill microbatches."""

        raise NotImplementedError

    @abstractmethod
    def propose_next_overlap(
        self,
        target_model_input0: ModelInput,  # batch_size = padded_verify_batch_size0
        target_model_output0: ModelOutput,  # logits: [padded_verify_batch_size0, vocab_size]
        target_next_token_ids0: torch.Tensor,  # [padded_verify_batch_size0]
        real_verify_rows0: int,
        accept_len0: torch.Tensor | None,  # [req_num0]
        target_model_input1: ModelInput,  # batch_size = padded_verify_batch_size1
        target_model_output1: ModelOutput,  # logits: [padded_verify_batch_size1, vocab_size]
        target_next_token_ids1: torch.Tensor,  # [padded_verify_batch_size1]
        real_verify_rows1: int,
        accept_len1: torch.Tensor | None,  # [req_num1]
        draft_step: int,
    ) -> SpecProposal:
        """Generate one proposal from two DP-overlapped decode microbatches."""

        raise NotImplementedError
