import inspect
import torch
from typing import List, Optional, Protocol, Tuple, runtime_checkable


def get_protocol_sampling_op_names(protocol: type) -> tuple[str, ...]:
    names: list[str] = []
    for name, value in protocol.__dict__.items():
        if name.startswith("_"):
            continue
        if inspect.isfunction(value):
            names.append(name)
    return tuple(names)


@runtime_checkable
class SamplingProtocol(Protocol):

    def apply_invalid_token_ids(
        self,
        *,
        logits: torch.Tensor,
        invalid_token_ids: torch.Tensor,
        cu_invalid_token_num: torch.Tensor,
    ) -> None: ...

    def apply_penalty(
        self,
        *,
        logits: torch.Tensor,
        b_req_idx: torch.Tensor,
        b_length_penalty_param: torch.Tensor,
        b_mask_eos_reqs: torch.Tensor,
        p_token_ids: torch.Tensor,
        p_token_counts: torch.Tensor,
        p_cumsum_seq_len: torch.Tensor,
        eos_ids: torch.Tensor,
        req_to_presence_penalty: torch.Tensor,
        req_to_frequency_penalty: torch.Tensor,
        req_to_repetition_penalty: torch.Tensor,
        req_to_exponential_decay_length_penalty: torch.Tensor,
        vocab_size: int,
    ) -> None: ...

    def apply_penalty_gpu_cache(
        self,
        *,
        logits: torch.Tensor,
        b_req_idx: torch.Tensor,
        b_length_penalty_param: torch.Tensor,
        b_mask_eos_reqs: torch.Tensor,
        eos_ids: torch.Tensor,
        req_to_presence_penalty: torch.Tensor,
        req_to_frequency_penalty: torch.Tensor,
        req_to_repetition_penalty: torch.Tensor,
        req_to_exponential_decay_length_penalty: torch.Tensor,
        req_to_out_token_id_counter: torch.Tensor,
        vocab_size: int,
    ) -> None: ...

    def top_p_top_k_sample(
        self,
        *,
        probs: torch.Tensor,
        top_ps: torch.Tensor,
        top_ks: torch.Tensor,
        generators: List[Optional[torch.Generator]] | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]: ...


SAMPLING_OP_NAMES: tuple[str, ...] = get_protocol_sampling_op_names(SamplingProtocol)
