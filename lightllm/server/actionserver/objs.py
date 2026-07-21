from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Optional

import torch

from lightllm.models.pi0.config import Pi0VLAConfig


class ActionStatus(IntEnum):
    IDLE = 0
    RUNNING = 1
    HAS_OUTPUT = 2
    ERROR = 3
    DONE = 4


class ActionOutcome(IntEnum):
    """Terminal result of one generation of an action branch."""

    NONE = 0
    SUCCESS = 1
    ERROR = 2
    TIMEOUT = 3
    ABORTED = 4
    RESTART_REQUIRED = 5


class ActionAckStatus(IntEnum):
    """Whether it is safe for the target runtime to recycle action resources."""

    NONE = 0
    WAITING_FOR_WORKERS = 1
    WORKERS_SAFE = 2
    SAFE_TO_RELEASE = 3
    RESTART_REQUIRED = 4


class ActionReleaseDecision(str, Enum):
    WAIT = "wait"
    RELEASE = "release"
    RESTART_REQUIRED = "restart_required"


class ActionContextOwnerDisposition(IntEnum):
    """HTTP disposition for a newly committed persistent context handle."""

    NONE = 0
    PENDING = 1
    DELIVERED = 2
    DISCARDED = 3


class ActionControlKind(str, Enum):
    TIMEOUT = "timeout"
    ABORT = "abort"


class PrefixContextOp(str, Enum):
    """Requested lifetime operation for a VLM prefix.

    ``ONESHOT`` preserves the original action-request contract. ``CREATE`` and
    ``REPLACE`` both require a target prefill, while ``REUSE`` consumes an
    already committed prefix and ``CLOSE`` is a control-only operation.
    """

    ONESHOT = "oneshot"
    CREATE = "create"
    REUSE = "reuse"
    REPLACE = "replace"
    CLOSE = "close"

    @property
    def requires_prefix_inputs(self) -> bool:
        return self in {
            PrefixContextOp.ONESHOT,
            PrefixContextOp.CREATE,
            PrefixContextOp.REPLACE,
        }

    @property
    def requires_existing_context(self) -> bool:
        return self in {
            PrefixContextOp.REUSE,
            PrefixContextOp.REPLACE,
            PrefixContextOp.CLOSE,
        }

    @property
    def requires_identity(self) -> bool:
        return self in {
            PrefixContextOp.REUSE,
            PrefixContextOp.REPLACE,
            PrefixContextOp.CLOSE,
        }

    @property
    def requires_context_id(self) -> bool:
        return self is PrefixContextOp.CREATE

    @property
    def produces_action(self) -> bool:
        return self is not PrefixContextOp.CLOSE


def _validate_protocol_id(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    if len(value) > 256:
        raise ValueError(f"{name} must contain at most 256 characters")


@dataclass(frozen=True)
class PrefixContextIdentity:
    """Concrete, restart-safe identity issued for a committed prefix.

    ``server_epoch`` invalidates handles across process restarts and ``version``
    makes copy-on-write replacement ABA-safe.
    """

    context_id: str
    version: int
    server_epoch: str

    def __post_init__(self):
        _validate_protocol_id(self.context_id, "prefix context_id")
        _validate_protocol_id(self.server_epoch, "prefix server_epoch")
        if isinstance(self.version, bool) or not isinstance(self.version, int):
            raise ValueError("prefix context version must be an integer")
        if self.version <= 0:
            raise ValueError("prefix context version must be positive")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PrefixContextIdentity":
        if not isinstance(value, dict):
            raise TypeError("prefix context identity must be an object")
        return cls(**value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_id": self.context_id,
            "version": self.version,
            "server_epoch": self.server_epoch,
        }


@dataclass(frozen=True)
class PrefixContextRef:
    """Operation plus an optional exact reference to a persistent prefix."""

    op: PrefixContextOp = PrefixContextOp.ONESHOT
    identity: Optional[PrefixContextIdentity] = None
    context_id: Optional[str] = None

    def __post_init__(self):
        op = self.op if isinstance(self.op, PrefixContextOp) else PrefixContextOp(self.op)
        identity = self.identity
        if isinstance(identity, dict):
            identity = PrefixContextIdentity.from_dict(identity)
        if identity is not None and not isinstance(identity, PrefixContextIdentity):
            raise TypeError("prefix context identity has an invalid type")
        context_id = self.context_id
        if context_id is not None:
            _validate_protocol_id(context_id, "prefix context_id")
        object.__setattr__(self, "op", op)
        object.__setattr__(self, "identity", identity)
        object.__setattr__(self, "context_id", context_id)

        if op.requires_identity and identity is None:
            raise ValueError(f"prefix context op={op.value} requires an identity")
        if not op.requires_identity and identity is not None:
            raise ValueError(f"prefix context op={op.value} must not carry an identity")
        if op.requires_context_id and context_id is None:
            raise ValueError("prefix context op=create requires a context_id")
        if not op.requires_context_id and context_id is not None:
            raise ValueError(f"prefix context op={op.value} must not carry a context_id")

    @classmethod
    def from_dict(cls, value: dict[str, Any] | str | None) -> "PrefixContextRef":
        if value is None:
            return cls()
        if isinstance(value, str):
            return cls(op=PrefixContextOp(value))
        if not isinstance(value, dict):
            raise TypeError("prefix_context must be an object")
        if "op" not in value and {
            "context_id",
            "version",
            "server_epoch",
        }.issubset(value):
            return cls(
                op=PrefixContextOp.REUSE,
                identity=PrefixContextIdentity.from_dict(value),
            )
        return cls(**value)

    def to_dict(self) -> dict[str, Any]:
        result = {"op": self.op.value}
        if self.identity is not None:
            result["identity"] = self.identity.to_dict()
        if self.context_id is not None:
            result["context_id"] = self.context_id
        return result


@dataclass(frozen=True)
class ActionTaskIdentity:
    """ABA-safe identity for a task occupying a reusable ShmReq slot."""

    slot_index: int
    generation: int
    task_id: int

    def __post_init__(self):
        if self.slot_index < 0:
            raise ValueError("action task slot_index must be non-negative")
        if self.generation < 0:
            raise ValueError("action task generation must be non-negative")
        if self.task_id < 0:
            raise ValueError("action task_id must be non-negative")


@dataclass
class ActionRequest:
    """Action-specific payload carried by the normal multimodal request."""

    state: Any
    request_id: Optional[str] = None
    raw_state: Any = None
    noise: Any = None
    action_horizon: Optional[int] = None
    action_dim: Optional[int] = None
    num_denoise_steps: Optional[int] = None
    timeout: float = 120.0
    metadata: dict[str, Any] = field(default_factory=dict)
    prefix_context: PrefixContextRef = field(default_factory=PrefixContextRef)

    def __post_init__(self):
        if not isinstance(self.prefix_context, PrefixContextRef):
            self.prefix_context = PrefixContextRef.from_dict(self.prefix_context)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ActionRequest":
        if not isinstance(value, dict):
            raise TypeError("action request must be an object")
        values = dict(value)
        if "prefix_context" in values:
            values["prefix_context"] = PrefixContextRef.from_dict(values["prefix_context"])
        return cls(**values)

    @property
    def context_op(self) -> PrefixContextOp:
        return self.prefix_context.op

    @property
    def context_identity(self) -> Optional[PrefixContextIdentity]:
        return self.prefix_context.identity

    @property
    def context_id(self) -> Optional[str]:
        identity = self.context_identity
        return identity.context_id if identity is not None else self.prefix_context.context_id

    @property
    def requires_prefix_inputs(self) -> bool:
        """Whether HTTP must provide prompt/images for a target prefill."""

        return self.context_op.requires_prefix_inputs

    @property
    def requires_existing_context(self) -> bool:
        return self.context_op.requires_existing_context

    @property
    def produces_action(self) -> bool:
        return self.context_op.produces_action

    @property
    def persists_prefix(self) -> bool:
        return self.context_op in {
            PrefixContextOp.CREATE,
            PrefixContextOp.REPLACE,
        }

    def accepts_model_state(self, config: Pi0VLAConfig) -> bool:
        """Whether ``state`` participates in this model operation.

        Pi0 consumes continuous state in every action task. Pi0.5 embeds state
        in a newly built VLM prefix, so an existing-prefix reuse cannot accept
        a replacement value.
        """

        return self.produces_action and not (config.is_pi05 and self.context_op is PrefixContextOp.REUSE)

    def requires_model_state(self, config: Pi0VLAConfig) -> bool:
        return self.accepts_model_state(config)

    def to_dict(self) -> dict[str, Any]:
        result = {
            "state": self.state,
            "request_id": self.request_id,
            "raw_state": self.raw_state,
            "noise": self.noise,
            "action_horizon": self.action_horizon,
            "action_dim": self.action_dim,
            "num_denoise_steps": self.num_denoise_steps,
            "timeout": self.timeout,
            "metadata": self.metadata,
        }
        # Preserve the legacy serialized shape for the default one-shot path.
        if self.context_op is not PrefixContextOp.ONESHOT:
            result["prefix_context"] = self.prefix_context.to_dict()
        return result

    def validate(self, config: Pi0VLAConfig) -> Pi0VLAConfig:
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("action timeout must be finite and positive")
        if not isinstance(self.metadata, dict):
            raise ValueError("action metadata must be an object")

        if not self.produces_action:
            control_fields = {
                "state": self.state,
                "raw_state": self.raw_state,
                "noise": self.noise,
                "action_horizon": self.action_horizon,
                "action_dim": self.action_dim,
                "num_denoise_steps": self.num_denoise_steps,
            }
            supplied = [name for name, value in control_fields.items() if value is not None]
            if supplied:
                raise ValueError("prefix context close must not carry action fields: " + ", ".join(supplied))
            return config

        # In pi0.5 the observation state is tokenized into the VLM prefix. A
        # state-only reuse would therefore silently ignore a new value; callers
        # must replace the prefix instead. ``raw_state`` remains available for
        # response post-processing and is intentionally not rejected here.
        if not self.accepts_model_state(config):
            if self.state is not None:
                raise ValueError("pi0.5 prefix reuse cannot update state; use replace to rebuild the prefix")
        elif self.state is None:
            raise ValueError(f"prefix context op={self.context_op.value} requires state")

        if self.state is not None:
            state = torch.as_tensor(self.state, dtype=torch.float32)
        else:
            state = None
        if state is not None and state.ndim not in {1, 2}:
            raise ValueError("state must have shape [state_dim] or [batch, state_dim]")
        if state is not None and state.ndim == 2 and state.shape[0] != 1:
            raise ValueError("the HTTP action path currently accepts one observation per request")
        if state is not None and state.shape[-1] == 0:
            raise ValueError("state must contain at least one value")
        if state is not None and state.shape[-1] > config.max_state_dim:
            raise ValueError(f"state_dim={state.shape[-1]} exceeds checkpoint maximum {config.max_state_dim}")
        if state is not None and not torch.isfinite(state).all():
            raise ValueError("state must contain only finite values")
        request_config = config.with_overrides(
            action_dim=self.action_dim,
            action_horizon=self.action_horizon,
            num_denoise_steps=self.num_denoise_steps,
        )
        if request_config.action_horizon > config.action_horizon:
            raise ValueError(
                f"action_horizon={request_config.action_horizon} exceeds the " f"server maximum {config.action_horizon}"
            )
        if request_config.num_denoise_steps > config.num_denoise_steps:
            raise ValueError(
                f"num_denoise_steps={request_config.num_denoise_steps} exceeds "
                f"the server maximum {config.num_denoise_steps}"
            )
        if self.noise is not None:
            noise = torch.as_tensor(self.noise, dtype=torch.float32)
            if noise.ndim == 2:
                noise = noise.unsqueeze(0)
            expected = (
                1,
                request_config.action_horizon,
                request_config.action_dim,
            )
            if tuple(noise.shape) != expected:
                raise ValueError(f"noise must have shape {expected}, got {tuple(noise.shape)}")
            if not torch.isfinite(noise).all():
                raise ValueError("noise must contain only finite values")
        return request_config


@dataclass
class ActionExpertTask:
    """Serializable action work item.

    Physical prefix and scratch mappings may be compact 1-D tensors for TP=1
    or rank-major tensors gathered by the target runtime for action TP.
    """

    request_id: int
    slot_index: int
    prefix_req_indexes: torch.Tensor
    # Persistent-context hot ticks omit both mappings after the first
    # successful task registered them in every action worker.
    prefix_mem_indexes: Optional[torch.Tensor]
    scratch_mem_indexes: Optional[torch.Tensor]
    prefix_seq_lens: torch.Tensor
    action_req_indexes: torch.Tensor
    state: Optional[torch.Tensor]
    noisy_actions: torch.Tensor
    action_horizon: int
    action_dim: int
    num_denoise_steps: int
    # ``generation`` must increase whenever ``slot_index`` is reused.  The
    # independent task id rejects duplicate delivery within one generation.
    generation: int
    task_id: int
    submitted_at: float = field(default_factory=time.perf_counter)
    # Uses the same monotonic clock as ``time.perf_counter``.
    deadline_at: Optional[float] = None
    ack_grace_seconds: float = 5.0
    prefix_context_identity: Optional[PrefixContextIdentity] = None

    @property
    def identity(self) -> ActionTaskIdentity:
        return ActionTaskIdentity(
            slot_index=self.slot_index,
            generation=self.generation,
            task_id=self.task_id,
        )

    @staticmethod
    def _select_rank_mapping(value: torch.Tensor, rank: int, name: str) -> torch.Tensor:
        if value.ndim == 1:
            return value
        if value.ndim not in {2, 3}:
            raise ValueError(f"{name} must be one-dimensional or have a leading TP-rank axis")
        if not 0 <= rank < value.shape[0]:
            raise IndexError(f"{name} has no mapping for action TP rank {rank}")
        return value[rank]

    def mappings_for_rank(self, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Select physical mappings for an action TP rank.

        One-dimensional tensors are used for TP=1. The runtime may dispatch
        rank-major ``[action_tp, length]`` tensors after gathering
        each target rank's allocator-specific physical indexes.  Prefix maps
        may additionally be ``[action_tp, batch, padded_prefix_length]``.
        """

        if self.prefix_mem_indexes is None or self.scratch_mem_indexes is None:
            raise RuntimeError("action task relies on a previously registered prefix context")
        prefix = self._select_rank_mapping(self.prefix_mem_indexes, rank, "prefix_mem_indexes")
        scratch = self._select_rank_mapping(self.scratch_mem_indexes, rank, "scratch_mem_indexes")
        return prefix, scratch


@dataclass(frozen=True)
class ActionPrefixContextRelease:
    """Forget worker-local mapping metadata after target KV is retired."""

    identity: PrefixContextIdentity


@dataclass(frozen=True)
class ActionControlRequest:
    """Abort/timeout request sent over the action manager's control channel."""

    slot_index: int
    generation: int
    task_id: int
    kind: ActionControlKind
    reason: str = ""
    grace_seconds: float = 5.0

    @property
    def identity(self) -> ActionTaskIdentity:
        return ActionTaskIdentity(
            slot_index=self.slot_index,
            generation=self.generation,
            task_id=self.task_id,
        )


@dataclass
class ActionWorkerAck:
    """An explicit lease-release acknowledgement from one action TP rank."""

    slot_index: int
    generation: int
    task_id: int
    rank: int
    outcome: ActionOutcome
    safe_to_release: bool
    actions: Optional[torch.Tensor] = None
    action_expert_ms: float = 0.0
    total_ms: float = 0.0
    error_info: Optional[str] = None

    @property
    def identity(self) -> ActionTaskIdentity:
        return ActionTaskIdentity(
            slot_index=self.slot_index,
            generation=self.generation,
            task_id=self.task_id,
        )


@dataclass
class ActionResponse:
    request_id: Optional[str]
    actions: Optional[torch.Tensor]
    action_horizon: int
    action_dim: int
    policy_timing: dict[str, float] = field(default_factory=dict)
    error_info: Optional[str] = None
    outcome: ActionOutcome = ActionOutcome.SUCCESS
    restart_required: bool = False
    prefix_context_version: Optional[int] = None
    # Internal delivery token. It is intentionally omitted from ``to_dict``
    # and only bridges the shared result slot to the public HTTP owner ACK.
    context_owner_identity: Optional[ActionTaskIdentity] = field(
        default=None,
        repr=False,
    )

    def to_dict(self) -> dict[str, Any]:
        result = {
            "request_id": self.request_id,
            "actions": None if self.actions is None else self.actions.cpu().tolist(),
            "action_horizon": self.action_horizon,
            "action_dim": self.action_dim,
            "policy_timing": self.policy_timing,
            "finish_status": "error" if self.error_info else "finished",
            "error_info": self.error_info,
            "action_outcome": self.outcome.name.lower(),
            "restart_required": self.restart_required,
        }
        if self.prefix_context_version is not None:
            result["prefix_context_version"] = self.prefix_context_version
        return result
