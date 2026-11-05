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
class StateAcceptance:
    """One accepted prefix per distinct request, in forward order, including the base input token."""

    # 接受结果是状态提交的唯一输入；batches 只携带对应 forward 的执行配置，避免调用方
    # 重新拼装请求顺序、接受长度和 microbatch 分区。

    req_indices: torch.Tensor
    accepted_lengths: torch.Tensor
    batches: Tuple[StateCommitBatch, ...] = ()

    def validate(self, state_names=None):
        if self.req_indices.ndim != 1 or self.accepted_lengths.shape != self.req_indices.shape:
            raise ValueError("request indices and accepted lengths must share one-dimensional layout")
        for value in (self.req_indices, self.accepted_lengths):
            if value.dtype not in (torch.int32, torch.int64) or value.device != self.req_indices.device:
                raise ValueError("acceptance tensors must be integer tensors on the same device")
        if self.batches and sum(batch.request_count for batch in self.batches) != self.req_indices.numel():
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
                yield self.req_indices[start:end], payload
            start = end
