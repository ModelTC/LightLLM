from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import torch

if TYPE_CHECKING:
    from lightllm.common.req_manager.req_sampling_params import ReqSamplingParamsManager
    from lightllm.server.router.model_infer.infer_batch import InferReq


class ApplyPenalty(Protocol):
    def __call__(
        self,
        Logits: torch.Tensor,
        b_req_idx: torch.Tensor,
        b_length_penalty_param: torch.Tensor,
        b_mask_eos_reqs: torch.Tensor,
        p_token_ids: torch.Tensor,
        p_token_counts: torch.Tensor,
        p_cumsum_seq_len: torch.Tensor,
        eos_ids: torch.Tensor,
        sampling_params_manager: "ReqSamplingParamsManager",
    ) -> None: ...


class ApplyPenaltyGpuCache(Protocol):
    def __call__(
        self,
        Logits: torch.Tensor,
        b_req_idx: torch.Tensor,
        b_length_penalty_param: torch.Tensor,
        b_mask_eos_reqs: torch.Tensor,
        eos_ids: torch.Tensor,
        sampling_params_manager: "ReqSamplingParamsManager",
    ) -> None: ...


class ApplyInvalidTokenIds(Protocol):
    def __call__(
        self,
        Logits: torch.Tensor,
        invalid_token_ids: torch.Tensor,
        cu_invalid_token_num: torch.Tensor,
    ) -> None: ...


class TokenIdCounter(Protocol):
    def __call__(
        self,
        prompt_ids: torch.Tensor,
        out_token_id_counter: torch.Tensor,
    ) -> None: ...


class UpdateReqToTokenIdCounter(Protocol):
    def __call__(
        self,
        b_req_idx: torch.Tensor,
        next_token_ids: torch.Tensor,
        req_to_out_token_id_counter: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> None: ...


class TopPTopKSample(Protocol):
    def __call__(
        self,
        reqs: list["InferReq"],
        probs: torch.Tensor,
        b_top_ps: torch.Tensor,
        b_top_ks: torch.Tensor,
        exist_req_use_random_seed: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


@dataclass(frozen=True)
class SamplingOps:
    apply_penalty: ApplyPenalty
    apply_penalty_gpu_cache: ApplyPenaltyGpuCache
    apply_invalid_token_ids: ApplyInvalidTokenIds
    token_id_counter: TokenIdCounter
    update_req_to_token_id_counter: UpdateReqToTokenIdCounter
    top_p_top_k_sample: TopPTopKSample
