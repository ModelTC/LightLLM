"""Wire objects shared by Action HTTP, orchestration, and model workers."""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Optional

import torch

from lightllm.server.activation_transfer import ActivationKind, ActivationRef, KVActivation


class ActionOutcome(IntEnum):
    """Terminal result of one generation of an action branch."""

    SUCCESS = 0
    ERROR = 1
    TIMEOUT = 2
    ABORTED = 3
    RESTART_REQUIRED = 4


@dataclass(frozen=True)
class ActionCancel:
    """Correlation-based cancellation, including context fast-path tasks."""

    correlation_id: str
    reason: str = "action client disconnected"

    def __post_init__(self):
        _validate_protocol_id(self.correlation_id, "action correlation_id")


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
        op = (
            self.op
            if isinstance(self.op, PrefixContextOp)
            else PrefixContextOp(self.op)
        )
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
            raise ValueError(
                f"prefix context op={op.value} must not carry a context_id"
            )

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

    # Index into the reusable shared-memory request array.
    slot_index: int
    # ShmReq request_id currently occupying that slot; detects slot reuse.
    generation: int
    # Individual action invocation. It changes when a persistent prefix is reused.
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

    # Depending on the model contract, observation state is either sent to the
    # action model or tokenized into a newly built target prefix.
    state: Any
    # External I/O profile selecting camera/state/action transforms. A learned
    # domain token, if a model has one, remains an ordinary model input.
    domain: Optional[str] = None
    # Optional external request identifier returned unchanged to the caller.
    request_id: Optional[str] = None
    # State in robot coordinates, used only for output post-processing. When
    # omitted, ``state`` is also used as the post-processing reference.
    raw_state: Any = None
    # Optional deterministic initial diffusion noise. Its canonical shape is
    # [1, action_horizon, max_action_dim].
    noise: Any = None
    # Per-request output limits; validation prevents exceeding server maxima.
    action_horizon: Optional[int] = None
    action_dim: Optional[int] = None
    num_denoise_steps: Optional[int] = None
    # Seed is used only to create noise when ``noise`` is absent.
    seed: Optional[int] = None
    # Whole action deadline, including waiting for target Prefix KV.
    timeout: float = 120.0
    # One-shot, create, reuse, replace or close operation on a VLM prefix.
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
            values["prefix_context"] = PrefixContextRef.from_dict(
                values["prefix_context"]
            )
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
        return (
            identity.context_id
            if identity is not None
            else self.prefix_context.context_id
        )

    @property
    def requires_prefix_inputs(self) -> bool:
        """Whether HTTP must provide prompt/images for a target prefill."""

        return self.context_op.requires_prefix_inputs

    @property
    def produces_action(self) -> bool:
        return self.context_op.produces_action

    @property
    def persists_prefix(self) -> bool:
        return self.context_op in {
            PrefixContextOp.CREATE,
            PrefixContextOp.REPLACE,
        }

    def accepts_model_state(self, config: dict) -> bool:
        """Whether ``state`` participates in this model operation.

        Models may consume continuous state in the Action Expert or embed it
        into a newly built VLM prefix. Existing-prefix reuse cannot replace
        state in the latter case.
        """

        return self.produces_action and not (
            config["state_in_prefix"] and self.context_op is PrefixContextOp.REUSE
        )

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
        }
        if self.domain is not None:
            result["domain"] = self.domain
        if self.seed is not None:
            result["seed"] = self.seed
        # Keep the default one-shot wire payload compact.
        if self.context_op is not PrefixContextOp.ONESHOT:
            result["prefix_context"] = self.prefix_context.to_dict()
        return result

    def validate(self, config: dict) -> dict:
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("action timeout must be finite and positive")
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int)
        ):
            raise ValueError("action seed must be an integer")

        profile = None
        if self.domain is not None:
            if not isinstance(self.domain, str) or not self.domain:
                raise ValueError("action domain must be a non-empty string")
            profile = (config.get("domain_profiles") or {}).get(self.domain)
            if profile is None:
                raise ValueError(f"unsupported action domain={self.domain!r}")

        if not self.produces_action:
            control_fields = {
                "state": self.state,
                "raw_state": self.raw_state,
                "noise": self.noise,
                "action_horizon": self.action_horizon,
                "action_dim": self.action_dim,
                "num_denoise_steps": self.num_denoise_steps,
            }
            supplied = [
                name for name, value in control_fields.items() if value is not None
            ]
            if supplied:
                raise ValueError(
                    "prefix context close must not carry action fields: "
                    + ", ".join(supplied)
                )
            return config

        # When observation state is tokenized into the VLM prefix, state-only
        # reuse would silently ignore a new value. Callers must replace the
        # prefix instead. ``raw_state`` remains available for post-processing.
        if not self.accepts_model_state(config):
            if self.state is not None:
                raise ValueError(
                    "prefix reuse cannot update embedded state; use replace"
                )
        elif self.state is None:
            raise ValueError(
                f"prefix context op={self.context_op.value} requires state"
            )

        if self.state is not None:
            state = torch.as_tensor(self.state, dtype=torch.float32)
        else:
            state = None
        if state is not None and state.ndim not in {1, 2}:
            raise ValueError("state must have shape [state_dim] or [batch, state_dim]")
        if state is not None and state.ndim == 2 and state.shape[0] != 1:
            raise ValueError(
                "the HTTP action path currently accepts one observation per request"
            )
        if state is not None and state.shape[-1] == 0:
            raise ValueError("state must contain at least one value")
        if state is not None and profile is not None:
            expected_state_dim = int(profile["state_dim"])
            if state.shape[-1] != expected_state_dim:
                raise ValueError(
                    f"domain={self.domain!r} requires state_dim={expected_state_dim}, "
                    f"got {state.shape[-1]}"
                )
        if state is not None and state.shape[-1] > config["max_state_dim"]:
            raise ValueError(
                f"state_dim={state.shape[-1]} exceeds checkpoint maximum {config['max_state_dim']}"
            )
        if state is not None and not torch.isfinite(state).all():
            raise ValueError("state must contain only finite values")
        # Per-request overrides live in a copy so the process-wide model
        # limits remain immutable across concurrent requests.
        request_config = dict(config)
        for value, name in (
            (self.action_dim, "action_dim"),
            (self.action_horizon, "action_horizon"),
            (self.num_denoise_steps, "num_denoise_steps"),
        ):
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise ValueError(f"{name} must be a positive integer")
                request_config[name] = value
        if profile is not None:
            expected_action_dim = int(profile["action_dim"])
            if self.action_dim is None:
                request_config["action_dim"] = expected_action_dim
            if request_config["action_dim"] != expected_action_dim:
                raise ValueError(
                    f"domain={self.domain!r} requires action_dim={expected_action_dim}, "
                    f"got {request_config['action_dim']}"
                )
        if request_config["action_dim"] > request_config["max_action_dim"]:
            raise ValueError(
                f"action_dim must be in [1, {request_config['max_action_dim']}], "
                f"got {request_config['action_dim']}"
            )
        if request_config["action_horizon"] > config["action_horizon"]:
            raise ValueError(
                f"action_horizon={request_config['action_horizon']} exceeds the "
                f"server maximum {config['action_horizon']}"
            )
        if request_config["num_denoise_steps"] > config["num_denoise_steps"]:
            raise ValueError(
                f"num_denoise_steps={request_config['num_denoise_steps']} exceeds "
                f"the server maximum {config['num_denoise_steps']}"
            )
        if self.noise is not None:
            noise = torch.as_tensor(self.noise, dtype=torch.float32)
            if noise.ndim == 2:
                noise = noise.unsqueeze(0)
            expected = (
                1,
                request_config["action_horizon"],
                request_config["max_action_dim"],
            )
            if tuple(noise.shape) != expected:
                raise ValueError(
                    f"noise must have shape {expected}, got {tuple(noise.shape)}"
                )
            if not torch.isfinite(noise).all():
                raise ValueError("noise must contain only finite values")
        return request_config


@dataclass
class ActionTask:
    """Small control-plane message submitted to Action GenerationServer.

    The task deliberately carries no KV mapping. GenerationServer holds the
    referenced ``ShmReq`` until the action consumer is done and resolves the
    target request row only after every target rank has published KV-ready.
    Persistent-context reuse uses the mapping retained by VLAServer and can
    therefore enter the ready queue immediately.
    """

    # Typed reference to the cold VLM activation. No tensor is serialized.
    activation: ActivationRef
    action_request: ActionRequest | dict[str, Any]
    # Request seed copied by HTTP so the transfer lease stays data-only.
    default_seed: int = -1
    # Action invocation identity; defaults to the cold request generation.
    task_id: Optional[int] = None
    # Transport-level identity used to match cancellation and the async reply.
    correlation_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    # Monotonic timestamp used to preserve the original HTTP deadline.
    submitted_at: float = field(default_factory=time.perf_counter)

    def __post_init__(self):
        if not isinstance(self.activation, ActivationRef):
            raise TypeError("action activation must be an ActivationRef")
        if self.activation.kind is not ActivationKind.KV:
            raise ValueError("the current action backend requires a KV activation")
        if isinstance(self.action_request, dict):
            self.action_request = ActionRequest.from_dict(self.action_request)
        if not isinstance(self.action_request, ActionRequest):
            raise TypeError("action_request must be an ActionRequest")
        if isinstance(self.default_seed, bool) or not isinstance(
            self.default_seed, int
        ):
            raise ValueError("action default_seed must be an integer")
        if self.task_id is None:
            self.task_id = self.activation.generation
        if self.task_id < 0:
            raise ValueError("action task_id must be non-negative")
        _validate_protocol_id(self.correlation_id, "action correlation_id")

    @property
    def identity(self) -> ActionTaskIdentity:
        return ActionTaskIdentity(
            slot_index=self.activation.slot_index,
            generation=self.activation.generation,
            task_id=int(self.task_id),
        )


@dataclass
class ActionContextTask:
    """High-frequency task that reuses or closes an ACTIVE prefix context."""

    action_request: ActionRequest | dict[str, Any]
    # Every reuse gets a new invocation ID while retaining the same Prefix KV owner.
    task_id: int = field(default_factory=lambda: uuid.uuid4().int & ((1 << 63) - 1))
    correlation_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    submitted_at: float = field(default_factory=time.perf_counter)

    def __post_init__(self):
        if isinstance(self.action_request, dict):
            self.action_request = ActionRequest.from_dict(self.action_request)
        if not isinstance(self.action_request, ActionRequest):
            raise TypeError("action_request must be an ActionRequest")
        if self.action_request.context_op not in {
            PrefixContextOp.REUSE,
            PrefixContextOp.CLOSE,
        }:
            raise ValueError("ActionContextTask only supports reuse or close")
        if self.task_id < 0:
            raise ValueError("action context task_id must be non-negative")
        _validate_protocol_id(self.correlation_id, "action correlation_id")


@dataclass
class ActionModelTask:
    """Prepared data-plane task sent from VLAServer to each action rank."""

    identity: ActionTaskIdentity
    # Read-only row and length of the target VLM Prefix KV.
    prefix_kv: KVActivation
    # Normalized [1, state_dim] tensor, or None for a prefix-state policy.
    state: Optional[torch.Tensor]
    # Initial diffusion sample [1, action_horizon, max_action_dim].
    noisy_actions: torch.Tensor
    # Public result shape. noisy_actions may be wider than action_dim.
    action_horizon: int
    action_dim: int
    num_denoise_steps: int

    def __post_init__(self):
        if not isinstance(self.prefix_kv, KVActivation):
            raise TypeError("prefix_kv must be a KVActivation")
        if (
            self.prefix_kv.shm_index != self.identity.slot_index
            or self.prefix_kv.request_id != self.identity.generation
        ):
            raise ValueError("action task and prefix owner identities differ")
        if self.action_horizon <= 0 or self.action_dim <= 0:
            raise ValueError("action shape must be positive")
        if self.num_denoise_steps <= 0:
            raise ValueError("num_denoise_steps must be positive")


@dataclass
class ActionResult:
    """Terminal VLAServer reply delivered directly to the submitting DEALER."""

    correlation_id: str
    request_id: Optional[str]
    outcome: ActionOutcome
    actions: Optional[torch.Tensor] = None
    action_horizon: int = 0
    action_dim: int = 0
    policy_timing: dict[str, float] = field(default_factory=dict)
    error_info: Optional[str] = None
    prefix_context: Optional[PrefixContextIdentity] = None
    # Internal delivery ownership: only a cold CREATE/REPLACE result may
    # abandon a newly committed context.  REUSE/CLOSE also carry the identity
    # but must never transfer its ownership to this individual response.
    committed_new_context: bool = False
    restart_required: bool = False

    @property
    def ok(self) -> bool:
        return self.outcome is ActionOutcome.SUCCESS and self.error_info is None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "request_id": self.request_id,
            "actions": None if self.actions is None else self.actions.cpu().tolist(),
            "action_horizon": self.action_horizon,
            "action_dim": self.action_dim,
            "policy_timing": self.policy_timing,
            "finish_status": "finished" if self.ok else "error",
            "error_info": self.error_info,
            "action_outcome": self.outcome.name.lower(),
            "restart_required": self.restart_required,
        }
        if self.prefix_context is not None:
            result["prefix_context"] = self.prefix_context.to_dict()
        return result


@dataclass
class ActionWorkerAck:
    """An explicit lease-release acknowledgement from one action TP rank."""

    # These three fields must match the submitted ActionTaskIdentity exactly.
    slot_index: int
    generation: int
    task_id: int
    # TP rank that retired its CUDA reads of the shared Prefix KV.
    rank: int
    outcome: ActionOutcome
    # True only after this rank has synchronized all Prefix KV reads.
    safe_to_release: bool
    # Rank 0 alone returns actions; other ranks return completion ACKs only.
    actions: Optional[torch.Tensor] = None
    action_expert_ms: float = 0.0
    error_info: Optional[str] = None

    @property
    def identity(self) -> ActionTaskIdentity:
        return ActionTaskIdentity(
            slot_index=self.slot_index,
            generation=self.generation,
            task_id=self.task_id,
        )
