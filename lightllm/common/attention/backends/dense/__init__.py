from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DenseInputs:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    sliding_window: tuple[int, int] = (-1, -1)


@dataclass(frozen=True)
class DensePrefillBatch:
    req_to_token_indexs: torch.Tensor
    b_req_idx: torch.Tensor
    b_seq_len: torch.Tensor
    b_q_start_loc: torch.Tensor
    b_ready_cache_len: torch.Tensor
    max_q_seq_len: int


@dataclass(frozen=True)
class DenseDecodeBatch:
    req_to_token_indexs: torch.Tensor
    b_req_idx: torch.Tensor
    b_seq_len: torch.Tensor
    max_kv_seq_len: int
    max_len_in_batch: int
