from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch

from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
from lightllm.server.router.model_infer.mtp_speculative.proposers.base import SpecProposal

if TYPE_CHECKING:
    from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend


class BaseDpOverlapProposer(ABC):
    """双 microbatch DP-overlap proposer 的独立接口。"""

    def __init__(self, *, backend: "ModeBackend", enable_dynmaic_mtp: bool) -> None:
        self.backend = backend
        self.enable_dynmaic_mtp = bool(enable_dynmaic_mtp)

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
        target_model_input1: ModelInput,  # batch_size = padded_verify_batch_size1
        target_model_output1: ModelOutput,  # logits: [padded_verify_batch_size1, vocab_size]
        target_next_token_ids: torch.Tensor,  # [real_verify_rows0 + real_verify_rows1]
        real_verify_rows0: int,
        real_verify_rows1: int,
        accept_len: torch.Tensor,  # [real_req_num0 + real_req_num1]
        draft_step: int,
    ) -> SpecProposal:
        """Generate one proposal from two DP-overlapped decode microbatches."""

        raise NotImplementedError

    @staticmethod
    def _build_padded_next_token_ids(
        target_next_token_ids: torch.Tensor,
        batch_size: int,
        real_verify_rows: int,
        device: torch.device,
        source_start: int,
    ) -> torch.Tensor:
        """将合并后的真实 target token 拆分并补齐到一个 microbatch 的 verify shape。"""

        assert 0 <= real_verify_rows <= batch_size
        padded_token_ids = torch.zeros((batch_size,), dtype=torch.int64, device=device)
        if real_verify_rows > 0:
            padded_token_ids[:real_verify_rows].copy_(
                target_next_token_ids[source_start : source_start + real_verify_rows],
                non_blocking=True,
            )
        return padded_token_ids
