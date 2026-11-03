from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass(frozen=True)
class StateCommit:
    state_name: str
    payload: object


@dataclass(frozen=True)
class StateCommitBatch:

    request_count: int
    commits: Tuple[StateCommit, ...] = ()

    def __post_init__(self):
        if type(self.request_count) is not int or self.request_count < 0:
            raise ValueError("commit batch requires a nonnegative request count")
        names = [commit.state_name for commit in self.commits]
        if len(set(names)) != len(names):
            raise ValueError("duplicate state commit in a forward")


@dataclass(frozen=True)
class MtpCommitContext:

    b_req_mtp_start_loc: torch.Tensor
    b_req_idx: torch.Tensor
    b_mtp_index: torch.Tensor
    accepted_index: torch.Tensor
    verify_width: int
    batches: Tuple[StateCommitBatch, ...] = ()

    def validate(self, request_count: int, state_names=None):
        if type(self.verify_width) is not int or self.verify_width <= 0:
            raise ValueError("verification width must be positive")
        rows = (self.b_req_idx, self.b_mtp_index, self.accepted_index)
        if any(row.ndim != 1 or row.numel() != self.b_req_idx.numel() for row in rows):
            raise ValueError("verification row tensors must have the same one-dimensional layout")
        if self.b_req_mtp_start_loc.numel() != request_count:
            raise ValueError("verification and logical request counts differ")
        if self.batches and sum(batch.request_count for batch in self.batches) != request_count:
            raise ValueError("commit batches must partition all unpadded requests")
        for batch in self.batches:
            for commit in batch.commits:
                if state_names is not None and commit.state_name not in state_names:
                    raise ValueError(f"commit belongs to an unknown state: {commit.state_name}")

    def partitions(self, state_name: str):
        start = 0
        for batch in self.batches:
            end = start + batch.request_count
            if end > start:
                payload = next((c.payload for c in batch.commits if c.state_name == state_name), None)
                yield slice(start, end), payload
            start = end
