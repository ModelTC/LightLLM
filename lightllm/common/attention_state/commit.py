from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class RequestGenerationStamp:
    role: str
    request_indices: torch.Tensor
    mtp_indices: torch.Tensor
    request_generations: torch.Tensor
    verification_steps: torch.Tensor
    request_domain: torch.Tensor = field(repr=False, compare=False)

    @classmethod
    def stamp(cls, role, request_indices, mtp_indices, generations, verification_steps):
        requests = request_indices.clone()
        return cls(
            role,
            requests,
            mtp_indices.clone(),
            generations[requests.long()].clone(),
            verification_steps.clone(),
            generations,
        )

    @classmethod
    def concatenate(cls, commits):
        if not commits or any(commit.role != commits[0].role for commit in commits):
            raise ValueError("MTP outputs must belong to the same model role")
        if any(commit.request_domain is not commits[0].request_domain for commit in commits):
            raise ValueError("MTP outputs must belong to the same request manager")
        return cls(
            commits[0].role,
            *(
                torch.cat([getattr(commit, name) for commit in commits])
                for name in ("request_indices", "mtp_indices", "request_generations", "verification_steps")
            ),
            commits[0].request_domain,
        )

    def validate_generation(self, generations):
        if generations is not self.request_domain:
            raise ValueError("MTP output belongs to a different request manager")
        torch._assert_async(
            torch.all(generations[self.request_indices.long()] == self.request_generations),
            "MTP output refers to a released or reused request slot",
        )

    def consume(self, req_manager):
        if self.role != "main":
            raise ValueError("target verification cannot commit a draft model output")
        self.validate_generation(req_manager.request_generations)
        indexes = self.request_indices.long()
        torch._assert_async(
            torch.all(
                (indexes == req_manager.HOLD_REQUEST_ID)
                | (
                    (self.verification_steps == req_manager.request_verify_steps[indexes])
                    & (self.verification_steps > req_manager.request_verified_steps[indexes])
                )
            ),
            "MTP output was already consumed or belongs to a different decode step",
        )
        req_manager.request_verified_steps.scatter_reduce_(
            0, indexes, self.verification_steps, reduce="amax", include_self=True
        )
