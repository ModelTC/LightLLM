from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BoundedStateCommit:
    role: str
    request_indices: torch.Tensor
    mtp_indices: torch.Tensor
    request_generations: torch.Tensor

    @classmethod
    def capture(cls, role, request_indices, mtp_indices, generations):
        requests = request_indices.clone()
        return cls(role, requests, mtp_indices.clone(), generations[requests.long()].clone())

    @classmethod
    def concatenate(cls, commits):
        if not commits or any(commit.role != commits[0].role for commit in commits):
            raise ValueError("MTP outputs must belong to the same model role")
        return cls(
            commits[0].role,
            *(
                torch.cat([getattr(commit, name) for commit in commits])
                for name in ("request_indices", "mtp_indices", "request_generations")
            ),
        )

    def validate_generation(self, generations):
        torch._assert_async(
            torch.all(generations[self.request_indices.long()] == self.request_generations),
            "MTP output refers to a released or reused request slot",
        )
