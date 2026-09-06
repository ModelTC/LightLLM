from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

import torch

from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput

if TYPE_CHECKING:
    from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend


@dataclass
class MtpMemIndexesToFree:
    """描述一组由 MTP proposal 持有、需要在 verify 后释放的临时 KV 索引。"""

    # 位于 CPU 上的临时 KV cache 索引。
    mem_indexes_cpu: torch.Tensor
    # 与 mem_indexes_cpu 形状一致的 bool Tensor；True 表示释放对应索引。
    # 为 None 时表示 mem_indexes_cpu 中的全部索引都需要释放。
    free_mask_cpu: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        if self.free_mask_cpu is None:
            return
        assert isinstance(self.free_mask_cpu, torch.Tensor)
        assert self.free_mask_cpu.dtype == torch.bool
        assert self.free_mask_cpu.shape == self.mem_indexes_cpu.shape


@dataclass
class SpecProposal:
    """Common candidate-token output produced by every MTP proposer.

    `token_ids` has shape `[req_num, draft_step]` and contains only draft-model
    candidates. The target model's latest token is kept separately and merged
    into the request-level MTP buffer only when the proposal is persisted.
    `extra_mem_indexes_cpu` uniformly tracks every KV slot considered for
    release, including rejected target rows and proposal-owned temporary rows.
    Mode-specific scheduling metadata belongs to the corresponding subclass.
    """

    token_ids: torch.Tensor
    extra_mem_indexes_cpu: List[MtpMemIndexesToFree] = field(default_factory=list)


class BaseSpecProposer(ABC):
    """Base class for algorithm-specific draft proposal generation.

    A proposer owns the draft-side state transition. The target model gives it
    the current target token ids plus captured target hidden features through
    the prefill-state and proposal hooks. The proposer returns candidate ids
    but does not verify acceptance; verification is handled by SpecEngine.
    """

    def __init__(self, *, backend: "ModeBackend", enable_dynmaic_mtp: bool) -> None:
        self.backend = backend
        self.enable_dynmaic_mtp = bool(enable_dynmaic_mtp)

    @abstractmethod
    def fill_draft_model_kv_state(
        self,
        target_model_input: ModelInput,
        target_model_output: ModelOutput,
        target_next_token_ids: torch.Tensor,
    ) -> None:
        """Build draft KV/state from target prefill before the first decode verify.

        Inputs:
        - `target_model_input`: target prompt ModelInput. Its request order and
          mem_indexes are reused by the draft state builder.
        - `target_model_output`: target output containing the features needed
          by the selected speculative algorithm.
        - `target_next_token_ids`: first accepted target token, shape [run_req_num].

        This hook only prepares draft-side state. It does not create proposal
        tokens; the first decode iteration creates them through `propose_next`.
        """

        raise NotImplementedError

    @abstractmethod
    def propose_next(
        self,
        target_model_input: ModelInput,  # batch_size = verify_batch_size
        target_model_output: ModelOutput,  # logits: [verify_batch_size, vocab_size]
        target_next_token_ids: torch.Tensor,  # [verify_batch_size]
        b_req_mtp_start_loc: torch.Tensor,  # [req_num]
        draft_step: int,
        accept_len: Optional[torch.Tensor] = None,  # [req_num]
    ) -> SpecProposal:
        """Generate candidate tokens after one target decode forward.

        `target_model_input` contains the target verify rows, possibly compacted
        by dynamic scheduling. `b_req_mtp_start_loc` identifies each logical
        request's first row.

        The returned proposal contains one dense row per logical request and
        only the `draft_step` candidate tokens produced by the draft model.
        """

        raise NotImplementedError
