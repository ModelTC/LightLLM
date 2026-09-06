from dataclasses import dataclass

import torch

from lightllm.server.router.model_infer.mtp_speculative.proposers.base import SpecProposal


@dataclass
class EagleSpecProposal(SpecProposal):
    """EAGLE proposal with optional selected-token probabilities."""

    schedule_scores: torch.Tensor | None = None


@dataclass
class VanillaSpecProposal(SpecProposal):
    """Vanilla proposal with optional selected-token probabilities."""

    schedule_scores: torch.Tensor | None = None


@dataclass
class DFlashSpecProposal(SpecProposal):
    """DFlash proposal with optional block-token probabilities."""

    schedule_scores: torch.Tensor | None = None


@dataclass
class DSparkSpecProposal(SpecProposal):
    """DSpark proposal with GPU confidence scores and their CPU planner view."""

    schedule_scores: torch.Tensor | None = None
    schedule_scores_cpu: torch.Tensor | None = None
