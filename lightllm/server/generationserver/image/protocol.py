"""Model-neutral wire objects for image generation.

The HTTP representation, LightLLM prefix ownership, and backend-specific
pipeline inputs deliberately stay separate.  These objects carry only stable
generation parameters and lightweight prefix handles between processes. Any
image model may implement the same contract.
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

from lightllm.server.activation_transfer import (
    ActivationKind,
    ActivationPayload,
    ActivationRef,
    HiddenStatesActivation,
    KVActivation,
)


class ImageOutcome(IntEnum):
    SUCCESS = 0
    ERROR = 1
    TIMEOUT = 2
    ABORTED = 3
    RESTART_REQUIRED = 4


@dataclass(frozen=True)
class ImageGenerationInput:
    """Model-neutral parameters for one generated image result."""

    prefix_roles: tuple[str, ...]
    generation_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    width: int = 1024
    height: int = 1024
    steps: int = 50
    guidance_scale: float = 4.0
    seed: int = 42
    num_outputs: int = 1
    cfg_interval: tuple[float, float] = (-1.0, 2.0)
    cfg_norm: str = "global"
    timestep_shift: float = 3.0
    timeout: float = 300.0

    def __post_init__(self):
        if (
            not self.prefix_roles
            or len(self.prefix_roles) != len(set(self.prefix_roles))
            or any(
                not isinstance(role, str) or not role or not role.isascii()
                for role in self.prefix_roles
            )
        ):
            raise ValueError("prefix_roles must contain unique non-empty ASCII strings")
        for value, name in (
            (self.generation_id, "generation_id"),
            (self.request_id, "request_id"),
        ):
            if not isinstance(value, str) or not value or not value.isascii():
                raise ValueError(f"{name} must be a non-empty ASCII string")
        for value, name in (
            (self.width, "width"),
            (self.height, "height"),
            (self.steps, "steps"),
            (self.num_outputs, "num_outputs"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.width % 32 or self.height % 32:
            raise ValueError("generation width and height must be divisible by 32")
        for value, name in (
            (self.guidance_scale, "guidance_scale"),
            (self.timestep_shift, "timestep_shift"),
            (self.timeout, "timeout"),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if len(self.cfg_interval) != 2 or any(
            not math.isfinite(v) for v in self.cfg_interval
        ):
            raise ValueError("cfg_interval must contain two finite values")
        if self.cfg_interval[0] > self.cfg_interval[1]:
            raise ValueError("cfg_interval must be ordered")
        if self.cfg_norm not in {"none", "global", "channel"}:
            raise ValueError("cfg_norm must be one of: none, global, channel")

    def to_dict(self) -> dict[str, Any]:
        return {
            "prefix_roles": list(self.prefix_roles),
            "generation_id": self.generation_id,
            "request_id": self.request_id,
            "width": self.width,
            "height": self.height,
            "steps": self.steps,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "num_outputs": self.num_outputs,
            "cfg_interval": list(self.cfg_interval),
            "cfg_norm": self.cfg_norm,
            "timestep_shift": self.timestep_shift,
            "timeout": self.timeout,
        }


@dataclass
class ImageActivationTask:
    """One activation owner submitted before its request reaches the router."""

    activation: ActivationRef
    role: str
    generation: ImageGenerationInput
    correlation_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    submitted_at: float = field(default_factory=time.perf_counter)

    def __post_init__(self):
        if not isinstance(self.activation, ActivationRef):
            raise TypeError("generation activation must be an ActivationRef")
        if (
            self.activation.kind is ActivationKind.KV
            and self.activation.position_offset is None
        ):
            raise ValueError("generation KV activation requires a position offset")
        if self.role not in self.generation.prefix_roles:
            raise ValueError(f"unexpected generation prefix role: {self.role}")
        if not self.correlation_id or not self.correlation_id.isascii():
            raise ValueError("generation correlation_id must be non-empty ASCII")


# Source compatibility for existing image-generation clients.
ImagePrefixTask = ImageActivationTask


@dataclass(frozen=True)
class ImageCancel:
    correlation_id: str
    reason: str = "generation request was cancelled"


@dataclass(frozen=True)
class ImageModelTask:
    generation: ImageGenerationInput
    activations: dict[str, ActivationPayload]

    def __post_init__(self):
        if set(self.activations) != set(self.generation.prefix_roles):
            raise ValueError(
                "generation model task does not contain the expected activations"
            )
        if not all(
            isinstance(value, (KVActivation, HiddenStatesActivation))
            for value in self.activations.values()
        ):
            raise TypeError("generation inputs must be activation descriptors")


@dataclass(frozen=True)
class GeneratedArtifact:
    mime_type: str
    data: str

    def __post_init__(self):
        if not self.mime_type or not isinstance(self.data, str):
            raise ValueError(
                "generated artifact must contain MIME type and encoded data"
            )

    def to_dict(self) -> dict[str, str]:
        return {
            "type": "image",
            "mime_type": self.mime_type,
            "b64_json": self.data,
        }


@dataclass
class ImageWorkerResult:
    generation_id: str
    rank: int
    outcome: ImageOutcome
    safe_to_release: bool
    artifacts: list[GeneratedArtifact] = field(default_factory=list)
    generation_ms: float = 0.0
    error_info: str | None = None


@dataclass
class ImageResult:
    correlation_id: str
    generation_id: str
    request_id: str
    outcome: ImageOutcome
    artifacts: list[GeneratedArtifact] = field(default_factory=list)
    generation_ms: float = 0.0
    error_info: str | None = None
    restart_required: bool = False

    @property
    def ok(self) -> bool:
        return self.outcome is ImageOutcome.SUCCESS and self.error_info is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "generation_id": self.generation_id,
            "data": [artifact.to_dict() for artifact in self.artifacts],
            "generation_ms": self.generation_ms,
            "finish_status": "finished" if self.ok else "error",
            "generation_outcome": self.outcome.name.lower(),
            "error_info": self.error_info,
            "restart_required": self.restart_required,
        }
